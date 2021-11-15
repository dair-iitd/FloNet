import copy

class Flowcharts(object):

    def __init__(self, flowchartJsons, glob=None):
        self._flowchartJsons = flowchartJsons
        if glob!=None:
            self._max_node_utterance_length = glob['max_node_utterance_length']
            self._max_edge_utterance_length = glob['max_edge_label_length']
        else:
            self._max_node_utterance_length = -1
            self._max_edge_utterance_length = -1

        self._populate_node_properties()

    @property
    def max_node_utterance_length(self):
        return self._max_node_utterance_length

    @property
    def max_edge_utterance_length(self):
        return self._max_edge_utterance_length
    
    def _populate_node_properties(self):
        self._paths_to_root = {}
        self._node_to_text_map  = {}
        for name, flowchart in self._flowchartJsons.items():
            parent_map = self._get_parents_map(flowchart)
            name = flowchart['name']
            self._paths_to_root[name] = {}
            self._node_to_text_map[name] = {}
            for node_id, node_properties in flowchart['nodes'].items():
                path_to_root = []
                curr_node = node_id
                while (curr_node in parent_map):
                    path_to_root.insert(0, parent_map[curr_node])
                    curr_node = parent_map[curr_node][0]
                self._paths_to_root[name][node_id] = copy.deepcopy(path_to_root)
                self._node_to_text_map[name][node_id] = node_properties['utterance']

    def _get_parents_map(self, flowchart):
        parent_map = {}
        for parent_node_id, edges in flowchart['edges'].items():
            for option, child_node_id in edges.items():
                parent_map[child_node_id] = (parent_node_id, option)
        return parent_map

    def get_node_text(self, flowchart, node_id):
        return self._node_to_text_map[flowchart][node_id]

    def get_all_node_ids(self, flowchart):
        return self._flowchartJsons[flowchart]['nodes'].keys() 

    def get_edge_from_parent(self, flowchart, node_id): 
        path_to_root = self._paths_to_root[flowchart][node_id]
        if len(path_to_root) == 0:
            return ""
        else:
            prev_node_edge_pair = path_to_root[-1]
            return prev_node_edge_pair[1]
    
    def get_parent_node_text(self, flowchart, node_id): 
        path_to_root = self._paths_to_root[flowchart][node_id]
        if len(path_to_root) == 0:
            return ""
        else:
            prev_node_edge_pair = path_to_root[-1]
            return self._node_to_text_map[flowchart][prev_node_edge_pair[0]]

    def get_full_flowchart(self, flowchart):
        return self._flowchartJsons[flowchart]

    def get_path_to_node(self, flowchart, node_id):
        return self._paths_to_root[flowchart][node_id]

    def get_flowchart_names(self):
        return list(self._flowchartJsons.keys())

    def get_all_paths(self, flowchart):
        return self._paths_to_root[flowchart]
