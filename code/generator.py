# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
import time
import numpy as np
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from nltk.tokenize import word_tokenize

import torch
from torchtext.data.metrics import bleu_score
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                  GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME, GPT2LMHeadModel, OpenAIGPTLMHeadModel)
from ignite.handlers import Timer

from gpt.gpt_utils import get_dataset, make_logdir, download_pretrained_model, pad_dataset
from gpt.gpt_utils import add_special_tokens_, get_data_loaders, average_distributed_scalar

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]

logger = logging.getLogger(__file__)
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../data/saved_data/flowchart_dummy_data.json", help="Path or url of the dataset. ")
    parser.add_argument("--dataset_cache", type=str, default='../data/saved_data/flowchart_dummy_data_cache', help="Path or url of the dataset cache")
    parser.add_argument("--use_flowchart", type=int, default=1, help="Use flowchart or not")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--log_path", type=str, default="../data/generator/", help="Path, url or short name of the model")
    parser.add_argument("--model", type=str, default="gpt2", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--suffix", type=str, default="", help="Suffix for folder that saves model checkpoint and logs")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=20, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")

    #for getting results
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")

    parser.add_argument("--sample_turns", type=int, default=10, help="number of time to sample for next token (repeated sampling if token is a special token)")
    parser.add_argument("--max_input_length", type=int, default=800, help="length of acceptable input. 512 for transfertransfero and 1024 for gpt2")
    parser.add_argument("--personality_length", type=int, default=200, help="length of acceptable flowchart input segment")
    parser.add_argument("--history_length", type=int, default=600, help="length of acceptable history input segment")
    parser.add_argument("--save_metric", type=str, default='BLEU', help="nll or BLEU")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))
    args.use_flowchart = True if args.use_flowchart==1 else False

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    return args

def init_model(args):
    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    if args.model_checkpoint == "":
        if args.model == 'gpt2':
            args.model_checkpoint = 'gpt2' #TODO fix this to a pretrained checkpoint
            #raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args.model_checkpoint = download_pretrained_model()
            #print(args.model_checkpoint)

    tokenizer_class = GPT2Tokenizer if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer # cant use Autotokenizer because checkpoint could be a Path
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)


    model_class = GPT2DoubleHeadsModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    # Add special tokens if they are not already added
    add_special_tokens_(model, tokenizer)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    #if args.fp16:
    #    from apex import amp  # Apex is only required if we use fp16 training
    #    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    return model, optimizer, tokenizer

##interact
import copy
import random
from gpt.gpt_utils import sample_sequence
import torch.nn.functional as F
import warnings



args = parse_args()
log_dir_path = args.log_path+"_".join([args.suffix,args.model,args.dataset_cache.split("/")[-1],args.save_metric])+"_"+str(int(time.time()))
model, optimizer, tokenizer = init_model(args)
logger.info("Prepare datasets")
if not args.use_flowchart:
    train_loader, valid_loader, test_loader, train_sampler, valid_sampler, test_sampler, valid_gen_loader, test_gen_loader, valid_gen_sampler, test_gen_sampler = get_data_loaders(args, tokenizer,use_flowchart=False)#TODO fix this to personality one
else:
    train_loader, valid_loader, test_loader, train_sampler, valid_sampler, test_sampler, valid_gen_loader, test_gen_loader, valid_gen_sampler, test_gen_sampler = get_data_loaders(args, tokenizer)#TODO fix this to personality one

if not os.path.exists(args.log_path):
    os.mkdir(args.log_path)

print('done')

# Training function and trainer
def update(engine, batch):
    model.train()
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
    input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, *_ = batch
    (lm_loss), (mc_loss), *_ = model(
        input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
        mc_labels=mc_labels, lm_labels=lm_labels
    )
    loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef) / args.gradient_accumulation_steps
    if args.fp16:
        raise ValueError("not implemented fp16")
        #with amp.scale_loss(loss, optimizer) as scaled_loss:
        #    scaled_loss.backward()
        #torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
    if engine.state.iteration % args.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()
trainer = Engine(update)

# Evaluation function and evaluator (evaluator output is the input of the metrics)
def inference(engine, batch):
    model.eval()
    with torch.no_grad():
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        #logger.info(tokenizer.decode(input_ids[0, -1, :].tolist(),skip_special_tokens=True))
        # if we dont send labels to model, it doesnt return losses
        lm_logits, mc_logits, *_ = model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
        )
        lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
        lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
        return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)

def inference_for_ppl(engine, batch):
    model.eval()
    with torch.no_grad():
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        #logger.info(tokenizer.decode(input_ids[0, -1, :].tolist(),skip_special_tokens=True))
        # if we dont send labels to model, it doesnt return losses
        lm_logits, mc_logits, *_ = model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
        )
        lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
        lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

        fct_loss_sum = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
        fct_loss_mean = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss_sum = fct_loss_sum(lm_logits_flat_shifted, lm_labels_flat_shifted)
        loss_mean = fct_loss_mean(lm_logits_flat_shifted, lm_labels_flat_shifted)
        global_loss_sum.append(loss_sum.item())
        global_loss_count.append(int(loss_sum/loss_mean))
        return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)

evaluator = Engine(inference)
tester = Engine(inference_for_ppl)

# Evaluation function and evaluator (evaluator output is the input of the metrics)
def infer_sequence(engine, batch):#TODO
    model.eval()
    with torch.no_grad():
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        valid_response, persona, history, n_persona, n_history, persona_lengths, history_lengths = batch
        #logger.info(tokenizer.decode(valid_response[0,:].tolist(),skip_special_tokens=True))
        outputs = []
        references = []
        persona, history = persona.tolist(), history.tolist()

        for i in range(valid_response.shape[0]):
            #remove paddings
            persona_, persona_lengths_ = persona[i][:n_persona[i]], persona_lengths[i][:n_persona[i]]
            history_, history_lengths_ = history[i][:n_history[i]], history_lengths[i][:n_history[i]]
            persona_ = [x[:persona_lengths_[i]] for i,x in enumerate(persona_)]
            history_ = [x[:history_lengths_[i]] for i,x in enumerate(history_)]

            out_ids = sample_sequence(persona_, history_, tokenizer, model, args)
            reference_ = tokenizer.decode(valid_response[i],skip_special_tokens=True)
            output_ = tokenizer.decode(out_ids,skip_special_tokens=True)
            references.append([word_tokenize(reference_)])
            outputs.append(word_tokenize(output_))

            #print everything
            print_string = "[{\"personality\":["
            for p in persona_:
                print_string+="\"" + tokenizer.decode(p) + "\","
            if len(persona_)>0:
                print_string=print_string[:-1]
            print_string+="],\"history\":["
            for h in history_:
                print_string+="\"" + tokenizer.decode(h) + "\","
            if len(history_)>0:
                print_string=print_string[:-1]
            print_string+="],\"reference\":\"" + reference_ + "\""
            print_string+=",\"output\":\"" + output_ + "\"}],"

            #add to global arrays
            bleu_references.extend(references)
            bleu_candidates.extend(outputs)

            if sample_evaluator.last_event_name == Events.COMPLETED:
                with open(log_dir_path+"/output_"+str(trainer.state.epoch)+".json", "a",encoding="UTF-8") as f:
                    f.write(print_string)
        return 0
sample_evaluator = Engine(infer_sequence)
sample_tester = Engine(infer_sequence)

bleu_candidates = []
bleu_references = []
global_loss_sum = []
global_loss_count = []

def add_events(trainer, evaluator, tester, sample_evaluator, sample_tester, args, train_loader, val_loader, test_loader, train_sampler, valid_sampler, test_sampler, valid_gen_loader, test_gen_loader, valid_gen_sampler, test_gen_sampler, optimizer,logger,bleu_candidates, bleu_references, global_loss_sum, global_loss_count):
    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda _: torch.cuda.empty_cache())
    if args.save_metric == "nll":
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: tester.run(test_loader))
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: sample_tester.run(test_gen_loader))
    else:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: sample_evaluator.run(valid_gen_loader))
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: sample_tester.run(test_gen_loader))
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: tester.run(test_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][0], x[1][0])),
                "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
    metrics["ppl"] = MetricsLambda(math.exp, metrics["nll"])
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    if args.save_metric == "nll":
        for name, metric in metrics.items():
            metric.attach(evaluator, name)
        for name, metric in metrics.items():
            metric.attach(tester, name)
    else:
        def bleu_calculator(bleu_candidates,bleu_references):
            bleu = bleu_score(bleu_candidates, bleu_references)
            bleu_references.clear()
            bleu_candidates.clear()
            return bleu
        def ppl_calculator(global_loss_sum,global_loss_count):
            total_sum_ = np.sum(global_loss_sum)
            total_count_ = np.sum(global_loss_count)
            total_ppl = np.exp(total_sum_/total_count_)
            global_loss_sum.clear()
            global_loss_count.clear()
            return total_ppl
        MetricsLambda(bleu_calculator,bleu_candidates,bleu_references).attach(sample_evaluator,"BLEU")
        MetricsLambda(bleu_calculator,bleu_candidates,bleu_references).attach(sample_tester,"BLEU")
        MetricsLambda(ppl_calculator,global_loss_sum,global_loss_count).attach(tester,"Total_PPL")
        for name, metric in metrics.items():
            metric.attach(tester, name)

    #add timers
    timer1 = Timer(average=True)
    timer1.attach(trainer,start=Events.EPOCH_STARTED,resume=Events.ITERATION_STARTED,pause=Events.ITERATION_COMPLETED,step=Events.ITERATION_COMPLETED)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: logger.info("Avg Training Time = " + str(timer1.value())))
    if args.save_metric=="nll":
        timer2 = Timer(average=True)
        timer3 = Timer(average=True)
        timer2.attach(evaluator,start=Events.EPOCH_STARTED,resume=Events.ITERATION_STARTED,pause=Events.ITERATION_COMPLETED,step=Events.ITERATION_COMPLETED)
        timer3.attach(tester,start=Events.EPOCH_STARTED,resume=Events.ITERATION_STARTED,pause=Events.ITERATION_COMPLETED,step=Events.ITERATION_COMPLETED)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, lambda _: logger.info("Avg Valid Time = "+ str(timer2.value())))
        tester.add_event_handler(Events.EPOCH_COMPLETED, lambda _: logger.info("Avg Test Time = "+ str(timer3.value())))
    else:
        timer3 = Timer(average=True)
        timer4 = Timer(average=True)
        timer5 = Timer(average=True)
        timer3.attach(tester,start=Events.EPOCH_STARTED,resume=Events.ITERATION_STARTED,pause=Events.ITERATION_COMPLETED,step=Events.ITERATION_COMPLETED)
        timer4.attach(sample_evaluator,start=Events.EPOCH_STARTED,resume=Events.ITERATION_STARTED,pause=Events.ITERATION_COMPLETED,step=Events.ITERATION_COMPLETED)
        timer5.attach(sample_tester,start=Events.EPOCH_STARTED,resume=Events.ITERATION_STARTED,pause=Events.ITERATION_COMPLETED,step=Events.ITERATION_COMPLETED)
        tester.add_event_handler(Events.EPOCH_COMPLETED, lambda _: logger.info("Avg Test Time = "+ str(timer3.value())))
        sample_evaluator.add_event_handler(Events.EPOCH_COMPLETED, lambda _: logger.info("Avg Valid Time = "+ str(timer4.value())))
        sample_tester.add_event_handler(Events.EPOCH_COMPLETED, lambda _: logger.info("Avg Test Time = "+ str(timer5.value())))


    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        if args.save_metric == "nll":
            evaluator.add_event_handler(Events.COMPLETED, lambda _: logger.info("Validation: %s" % pformat(evaluator.state.metrics)))
        else:
            sample_evaluator.add_event_handler(Events.COMPLETED, lambda _: logger.info("Validation: %s" % pformat(sample_evaluator.state.metrics)))

        log_dir = log_dir_path
        tb_logger = TensorboardLogger(log_dir)

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        
        if args.save_metric == "nll":
            tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)
            tb_logger.attach(tester, log_handler=OutputHandler(tag="test", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)
        else:
            tb_logger.attach(sample_evaluator, log_handler=OutputHandler(tag="validation_sampling", metric_names=["BLEU"], another_engine=trainer), event_name=Events.EPOCH_COMPLETED)
            tb_logger.attach(sample_tester, log_handler=OutputHandler(tag="test_sampling", metric_names=["BLEU"], another_engine=trainer), event_name=Events.EPOCH_COMPLETED)
            tb_logger.attach(tester, log_handler=OutputHandler(tag="test_sampling", metric_names=["Total_PPL"], another_engine=trainer), event_name=Events.EPOCH_COMPLETED)
            tb_logger.attach(tester, log_handler=OutputHandler(tag="test", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        if args.save_metric == "nll":
            score_function = lambda engine:-engine.state.metrics['nll'] 
            checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=2, score_function= score_function,score_name=args.save_metric)
            evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation
        else:
            score_function = lambda engine:engine.state.metrics['BLEU'] 
            checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=2, score_function= score_function,score_name=args.save_metric)
            epoch_checkpoint_handler = ModelCheckpoint(log_dir, 'last_checkpoint', save_interval=1, n_saved=1)
            sample_evaluator.add_event_handler(Events.EPOCH_COMPLETED, epoch_checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation
            sample_evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation
 
        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

    return trainer, evaluator, tester, sample_evaluator, sample_tester, checkpoint_handler, log_dir, tb_logger
trainer, evaluator, tester, sample_evaluator, sample_tester, checkpoint_handler, log_dir, tb_logger = add_events(trainer, evaluator, tester, sample_evaluator, sample_tester, args, train_loader, valid_loader, test_loader, train_sampler, valid_sampler, test_sampler, valid_gen_loader, test_gen_loader, valid_gen_sampler, test_gen_sampler, optimizer, logger, bleu_candidates, bleu_references, global_loss_sum, global_loss_count)

# Run the training
trainer.run(train_loader, max_epochs=args.n_epochs)

# On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
if args.local_rank in [-1, 0] and args.n_epochs > 0:
    os.rename(os.path.join(log_dir, checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
    tb_logger.close()
    logger.info("checkpoint used: %s", checkpoint_handler._saved[-1][1])
    logger.info("Log Directory: %s", log_dir)
