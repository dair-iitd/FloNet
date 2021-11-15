import bcolz
import numpy as np
import pickle
import pathlib
import os

def save_glove(dim):
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir='./glove6B/6B.'+dim+'.dat', mode='w')

    with open('./glove6B/glove.6B.'+dim+'d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
        
    vectors = bcolz.carray(vectors[1:].reshape((400000, int(dim))), rootdir='./glove6B/6B.'+dim+'.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open('./glove6B/6B.'+dim+'_words.pkl', 'wb'))
    pickle.dump(word2idx, open('./glove6B/6B.'+dim+'_idx.pkl', 'wb'))

def load_glove(glove_path, dim):
    vectors = bcolz.open(f'{glove_path}/6B.'+dim+'.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.'+dim+'_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.'+dim+'_idx.pkl', 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}
    return glove

def save_embedding_matrix(glob_path,dim,glove_path,prefix):
    glove = load_glove(glove_path,dim)
    with open(glob_path,"rb") as f:
        _, _, _, glob = pickle.load(f)
    glob_words = glob['encoder_vocab_to_idx']
    matrix_len = len(glob_words)
    uncommon_words=[]
    weights_matrix = np.zeros((matrix_len, int(dim)))
    words_found = 0
    emb_dim = int(dim)
    missed_word_idxs = []
    for word, i in glob_words.items():
        try: 
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            uncommon_words.append(word)
            missed_word_idxs.append(i)
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
    #save
    matrix_path = glove_path+"/saved_embedding_matrix"+dim+"_"+prefix+".pkl"
    missed_idx_path = glove_path+"/new_embedding_idx_"+dim+"_"+prefix+".pkl"
    with open(matrix_path,"wb") as f:
        pickle.dump(weights_matrix,f)
    with open(missed_idx_path,"wb") as f:
        pickle.dump(missed_word_idxs,f)
    return missed_word_idxs


def cache_embedding_matrix(glob,dim,glove_path,prefix,glob_path,no_hard_refresh=False, refresh=False):
    print("WWWWWWWWWW")
    matrix_path = glove_path+"/saved_embedding_matrix"+str(dim)+"_"+prefix+".pkl"
    missed_idx_path = glove_path+"/new_embedding_idx_"+str(dim)+"_"+prefix+".pkl"
    print("matrix_path", matrix_path)
    print("missed_idx_path", missed_idx_path)
    if no_hard_refresh:
        return matrix_path, missed_idx_path
    #check if glob is newer than the saved matrix
    fname_glove = pathlib.Path(matrix_path)
    fname_blob = pathlib.Path(glob_path)

    exists = os.path.exists(matrix_path)
    if (not exists) or refresh:
        glove = load_glove("./glove6B/",str(dim))
        glob_words = glob['encoder_vocab_to_idx']
        matrix_len = len(glob_words)
        uncommon_words=[]
        weights_matrix = np.zeros((matrix_len, int(dim)))
        words_found = 0
        emb_dim = int(dim)
        missed_word_idxs = []
        for word, i in glob_words.items():
            try: 
                weights_matrix[i] = glove[word]
                words_found += 1
            except KeyError:
                uncommon_words.append(word)
                missed_word_idxs.append(i)
                weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
        #save
        with open(matrix_path,"wb") as f:
            pickle.dump(weights_matrix,f)
        with open(missed_idx_path,"wb") as f:
            pickle.dump([missed_word_idxs, uncommon_words],f)
    return matrix_path, missed_idx_path

if __name__ == "__main__":
    dim=str(100)
    glove_path = "../glove6B"
    prefix = "out_domain_dialogs"
    glob_path = '../../data/saved_data/cached_'+prefix+'_score_FAQ.pkl'

    #save_glove(dim)
    save_embedding_matrix(glob_path,dim,glove_path,prefix)