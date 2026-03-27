import os
import networkx as nx
import copy
import math
import itertools
import re
from src.misc import utils
from src.data_structures.abstract_state import AbstractState

import numpy as np

class CAT():

    class AbsNode():
        def __init__(self, id, split, abs_state):
            self._id = id
            self._split = split
            self._state = abs_state
            self._parent = None
            self._child = []
            self.max_vals = None 
            self.min_vals = None
            self.mask = None 
            self._split_on = {}   # dict of dict: v1 -> { (a,b): [2,4], (c,d):[3,5]}
            self.clf = None

        def get_normalized_state(self,con_state: np.ndarray): 
            return (con_state - self.min_vals) / (self.max_vals - self.min_vals)
        
        def __str__(self):
            return str(self._state) + "_" + str(self.id)

    def __init__(self, state_ranges, is_int_state_variable, allowed_diff_to_refine, flexible_refinement):
        self._leaves = {}
        self._id_to_node = {}
        self.state_ranges = state_ranges
        self.is_int_state_variable = is_int_state_variable
        self._allowed_diff_to_refine = allowed_diff_to_refine
        self._flexible_refinement = flexible_refinement
        self._root = self.initialize_root_node(_id = 0)
        self._n_abstract_nodes = 1
        self._n_abstract_states = 1
        
    #################################### Initialize and Update State Abstraction ############################
    def initialize_root_node(self, _id):
        # compute flattened list containing low, high alternately for each state variable        
        abs_state = []
        for sv in self.state_ranges:
            abs_state.append(utils.avoid_negative_zero(sv[0]))
            abs_state.append(utils.avoid_negative_zero(sv[1]))
        self._root_name = tuple(abs_state)
        print("Root abstract state: ", [str(item) for item in self._root_name])

        # create abs_state for root
        root_abs_state = AbstractState(self._root_name, _id)

        # compute list of [low, mid, high] for each state variable which is useful when refining this abstract state
        root_split = []
        for i in range (len(self.state_ranges)):
            # midpoint is rounded off to 3 decimals
            midpoint = np.float32(round(self.state_ranges[i][0] + (self.state_ranges[i][1] - self.state_ranges[i][0])/2.0, 3)) 
            midpoint = utils.avoid_negative_zero(midpoint)
            min_value = np.float32(self.state_ranges[i][0])
            max_value = np.float32(self.state_ranges[i][1])
            root_split.append([min_value, midpoint, max_value])

        # create node for root
        node = self.add_node(_id, root_split, root_abs_state)
        return node

    def add_node(self, id, split, abs_state):
        node  = self.AbsNode(id, split, abs_state)
        self._leaves[abs_state] = node
        self._id_to_node[id] = node
        return node

    def find_node(self, abs_state):
        if abs_state in self._leaves:
            return self._leaves[abs_state]
        else:
            return None

    def state_to_split_indices(self, state, split):
        indices = []
        for i in range(len(split)): 
            s = [state.state[2*i],state.state[2*i+1]]
            for j in range(len(split[i])-1):
                if np.allclose(s, split[i][j:j+2]):
                    indices.append(j)
                    break
        return indices

    def is_refinable(self, interval, is_int):
        if is_int:
            if math.floor(interval[1]) - math.ceil(interval[0]) < 1:
                return False
            else:
                return True    
        else:
            if interval[1] - interval[0] >= self._allowed_diff_to_refine: 
                return True 
            else: 
                return False

    def refine_state_uniform_partitioning(self, unstable_state, refine_variables=None):
        if refine_variables is None:
            refine_variables = [1]*(len(unstable_state.state)//2) 

        # first find the new split
        node = self.find_node(unstable_state)
        if node._parent is None: 
            split = self.state_ranges
            for i in range(len(split)): 
                split[i] = list(split[i])
        else: 
            split = node._parent._split
        new_split, new_variable_values = self.update_split(unstable_state, split, refine_variables)
        for i in range(len(refine_variables)):
            if refine_variables[i]==1:
                node._split_on[i] = {}
                for interval in new_variable_values[i]:
                    node._split_on[i][tuple(interval)] = []
        node._split = new_split # the node now has a different split compared to its parent 
        new_state_values = list(itertools.product(*new_variable_values))
        new_state_values = [tuple([x for subnode in new_state_value for x in subnode]) for new_state_value in new_state_values]
        del self._leaves[unstable_state] # the no longer is a leaf node

        # then create new abstract states
        new_abstract_states = []
        for s in new_state_values:
            self._n_abstract_nodes += 1
            abs_state = AbstractState(s, self._n_abstract_nodes-1)
            new_node = self.add_node(id = abs_state.id, split = [], abs_state = abs_state)
            new_node._parent = node
            node._child.append(new_node)
            for i in range(len(refine_variables)):
                if refine_variables[i] == 1:
                    node._split_on[i][(s[2*i], s[2*i+1])].append(new_node._id)
            new_abstract_states.append(abs_state)
        self.update_n_abstract_states()
        return new_abstract_states
    
    def refine_state_flexible_refinement(self, unstable_state, clf, mask, min_vals, max_vals, class_to_states, class_to_minm_maxm):
        new_abstract_states = []
        # if len(class_to_states) > 1:
        #     print(f"\tRefining state: {unstable_state}")
        node = self.find_node(unstable_state)
        if node is None:
            print("Abstract state to be refined not found in CAT leaves.")
            return new_abstract_states
        if unstable_state in self._leaves:
            del self._leaves[unstable_state]
        for i in range(len(class_to_states)):
            self._n_abstract_nodes += 1
            s = []
            for item in class_to_minm_maxm[i]:
                s.append(item[0])
                s.append(item[1])
            s = tuple(s)
            abstract_state = AbstractState(s, self._n_abstract_nodes-1)
            node.clf = clf
            node.mask = mask 
            node.max_vals = max_vals 
            node.min_vals = min_vals
            new_abstract_states.append(abstract_state)
            new_node = self.add_node(abstract_state.id, split = [], abs_state = abstract_state)
            new_node.clf = None
            new_node._parent = node
            node._child.append(new_node)
            self._leaves[abstract_state] = new_node
        return new_abstract_states

    def update_split(self, unstable_state, old_split, to_split_vector):
        # returns a new split=[[low, mid, high] for each state variable] for the abstract state being refined 
        # and the new values after refining the abstract state
        new_split = copy.deepcopy(old_split)
        split_indices = self.state_to_split_indices(unstable_state, new_split)
        if unstable_state.state == self._root_name: 
            split_indices = []
            for i in range (len(self.state_ranges)):
                split_indices.append(0)
        new_state_values = []
        for i in range(len(split_indices)):
            index = split_indices[i]
            # if we need to split the state variable
            if to_split_vector[i] == 1:
                if self.is_refinable([new_split[i][index], new_split[i][index+1]], self.is_int_state_variable[i]): # if the specific range is refinable
                    new_split_point = np.float32(round(new_split[i][index] + (new_split[i][index+1] - new_split[i][index])/2.0, 3)) 
                    new_split_point = utils.avoid_negative_zero(new_split_point)
                    new_split[i].append(new_split_point)
                    new_split[i].sort()
                    new_state_values.append([[new_split[i][index], new_split_point], [new_split_point, new_split[i][index + 2]]])
                else: 
                    new_state_values.append([[new_split[i][index], new_split[i][index + 1]]])
            else:
                new_state_values.append([[new_split[i][index], new_split[i][index + 1]]])
                new_split[i] = old_split[i]
        return new_split, new_state_values

    ############################## load a CAT #####################################
    
    def update_n_abstract_states(self):
        self._n_abstract_states = len(self._leaves)

    def convert_to_tuple(self, a):
        a = a.replace('(','')
        a = a.replace(')','')
        a = a.replace('\'','')
        a = a.split(",")
        a = [np.float32(x) for x in a]
        return tuple(a) 

    def is_root(self, graph, node):
        if graph.in_degree(node) == 0:
            return True
        return False

    def load_cat(self, reuse_cat_path):
        self._root._split = [[0,2],[0,1,2,3]]
        self._n_abstract_nodes = 1
        graph = nx.MultiDiGraph(nx.nx_pydot.read_dot(reuse_cat_path))
        label_mapping = {}
        state_ranges = []
        for node in graph.nodes:
            pattern = re.compile(r'\(.*\)')
            re_obj = re.search(pattern, node)
            if re_obj:
                state_tup = re_obj.group(0)
                state_tup = self.convert_to_tuple(state_tup)
                # if "root" in node:
                if self.is_root(graph, node):
                    root_state_tup = copy.deepcopy(state_tup)
                    for i in range(0,len(state_tup)-1,1):
                        new_item_list = [state_tup[i], state_tup[i+1]]
                        state_ranges.append(tuple(new_item_list))
                label_mapping[node] = state_tup
        graph = nx.relabel_nodes(graph, label_mapping)
        
        root_node = self._root
        mapping = {}
        mapping[root_state_tup] = root_node
        
        stack = [root_state_tup]
        visited = []
        leaves = []
        while stack:
            node1_tup = stack.pop()
            if node1_tup in visited:
                continue
            visited.append(node1_tup)
            
            children = [item[1] for item in list(graph.out_edges(node1_tup))]
            for node2_tup in children:
                if node1_tup == root_state_tup:
                    new_node1 = mapping[root_state_tup]
                else:
                    if node1_tup in mapping:
                        new_node1 = mapping[node1_tup]
                    else:
                        self._n_abstract_nodes += 1
                        abs_state = AbstractState(node1_tup, self._n_abstract_nodes-1)
                        new_node1 = self.add_node(id = abs_state.id, split = [], abs_state = abs_state)
                        mapping[node1_tup] = new_node1
                if node2_tup in mapping:
                    new_node2 = mapping[node2_tup]
                else:
                    self._n_abstract_nodes += 1
                    abs_state = AbstractState(node2_tup, self._n_abstract_nodes-1)
                    new_node2 = self.add_node(id = abs_state.id, split = [], abs_state = abs_state)
                    mapping[node2_tup] = new_node1
                
                if new_node1._state in self._leaves:
                    del self._leaves[new_node1._state]
                
                new_node1._child.append(new_node2)
                new_node2._parent = new_node1
                if new_node1._parent:
                    vector = list()
                    for i in range(len(new_node1._state)):
                        if new_node1._state[i] != new_node2._state[i]:
                            vector.append(1)
                        else:
                            vector.append(0)
                    new_node1._split, _ = self.update_split(new_node1._state, new_node1._parent._split, vector)
                mapping[node1_tup] = new_node1
                mapping[node2_tup] = new_node2
                
                if node2_tup not in visited:
                    stack.append(node2_tup)
            if len(children) == 0:
                leaves.append(mapping[node1_tup])
        self._leaves = {leaf._state: leaf for leaf in leaves}
        self._root = mapping[root_state_tup]
    
    ################################ Initialize and Update Action Abstraction ############################################
    def refine_action(self, unstable_action, is_int_variable):
        new_variable_values = []
        for var_i in range(len(unstable_action)):
            if self.is_refinable(unstable_action[var_i], is_int_variable):
                if is_int_variable:
                    midpoint = round((unstable_action[var_i][0] + unstable_action[var_i][1]) / 2.0)
                    midpoint = utils.avoid_negative_zero(midpoint)
                    new_variable_values.append([[unstable_action[var_i][0], midpoint+1], [midpoint+1, unstable_action[var_i][1]+1]])
                else:
                    midpoint = np.float32(round((unstable_action[var_i][0] + unstable_action[var_i][1]) / 2.0, 3))
                    midpoint = utils.avoid_negative_zero(midpoint)
                    new_variable_values.append([[unstable_action[var_i][0], midpoint], [midpoint, unstable_action[var_i][1]]])
            else:
                new_variable_values.append([unstable_action[var_i]])
        new_values = list(itertools.product(*new_variable_values))
        new_values = [list(new_state_value) for new_state_value in new_values]
        return new_values

    
    def fallsWithinNew(self, concrete_state, abstract_state, vars_i, maximum):
        concrete_state = np.array(concrete_state, np.float32)
        ranges = np.asarray(abstract_state, np.float32)
        maximum = np.asarray(maximum).take(vars_i)
        a = ranges[::2]
        b = ranges[1::2]
        c = np.equal(b,maximum)
        c = np.asarray(c, np.float32)*0.01
        b = np.add(b,c)

        if (concrete_state >= a).all() and (concrete_state < b).all():
            return True
        else:
            return False
    
        
    def state_recursive_general(self, con_state_action, start_node: AbsNode):
        if start_node.clf is None:
            if start_node._state in self._leaves:
                return start_node._state
            else:
                children = start_node._child
                maximum = [item[-1] for item in start_node._split]
                for AbsNode in children:
                    if self.fallsWithinNew(con_state_action, AbsNode._state.state, [i for i in range(len(con_state_action))], maximum):
                        start_node = AbsNode
        else:
            con_state_action = np.array(con_state_action, np.float32)
            con_state_action_reduced = con_state_action[start_node.mask]
            state_con_normalized = start_node.get_normalized_state(con_state_action_reduced)
            pred = start_node.clf.predict([state_con_normalized])[0]
            start_node = start_node._child[pred]
        return self.state_recursive_general(con_state_action, start_node)


    ############################## functions for visualization of CAT ##########################################
    def get_networkx_cat(self):
        graph = nx.DiGraph()
        stack = [self._root]
        while stack:
            temp = stack.pop()
            for child in temp._child:
                stack.append(child)
                node1 = temp._state
                node2 = child._state 
                graph.add_edge(node1, node2)
        return graph
    
    def relabel(self,old_node,gran): 
        new_label = []
        old_state = old_node.state
        for i in range(0, len(old_state), 2): 
            t = [old_state[i],old_state[i+1] ]
            for i in range(len(t)):
                t[i] = round(t[i] * gran,2)
            new_label.append(str(tuple(t)))
        return tuple(new_label)

    def plot_cat(self, directory, index, best_actions):
        graph = self.get_networkx_cat()

        for node in graph.nodes:
            if node.id==0: 
                graph.add_node(node)
            if self.find_node(node):
                graph.add_node(node, shape="box")

        mapping = dict()
        for node in graph.nodes:
            mapping[node] = str(node)
        graph = nx.relabel_nodes(graph,mapping)

        if not os.path.exists(directory+"/"):
            os.makedirs(directory+"/")
        nx.nx_pydot.write_dot(graph, directory+"/cat_"+str(index)+".dot")