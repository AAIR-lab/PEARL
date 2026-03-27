from collections import OrderedDict
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import os
import math
import itertools
import numpy as np
import cv2
import json
from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in scalar divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom <= 0')
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.data_structures.cat import *
from src.data_structures.abstract_action import *
from src.abstraction.flexible_refinement import *
from src.misc import utils
from src.misc.visualize import VisualizeAbstraction

class Abstraction:
    def __init__(
            self, 
            seed,
            env, 
            agent,
            agent_con,
            k_cap,
            k_cap_actions,
            bootstrap = "from_init", #'from_concrete' 'from_ancestor' 'from_init'      
            refinement_method = "aggressive",
            maximum_state_variables_to_split = 1,
            init_action_abs_level = 1,
            init_state_abs_level = 1,
            flexible_refinement = False,
            max_clusters = 4,
            min_samples = 10,
            kernel = "linear",
            directory="",
            allowed_diff_to_refine = 0.1,
            fixed_cat = False,
            reuse_cat_path = None,
            plot_abstractions = False,
            init_beta = 1.0, 
            decay_beta_amount = 0.05, 
            min_beta = 0.15
            ):
        
        self._env = env
        self._agent = agent
        self._agent_con = agent_con
        self._k_cap = k_cap
        self._k_cap_actions = k_cap_actions
        self._bootstrap = bootstrap 
        self._refinement_method = refinement_method
        self._maximum_state_variables_to_split = maximum_state_variables_to_split
        self._init_action_abs_level = init_action_abs_level
        self._init_state_abs_level = init_state_abs_level
        self._flexible_refinement = flexible_refinement
        self._max_clusters = max_clusters
        self._min_samples = min_samples
        self.kernel = kernel
        self._directory = directory
        self._allowed_diff_to_refine = allowed_diff_to_refine
        self._fixed_cat = fixed_cat
        self._plot_abstractions = plot_abstractions
        self._is_action_space_discrete = self._env.is_action_space_discrete
        self.abstraction_directory = "abstraction"
        self.reuse_cat_path = reuse_cat_path

        self._init_beta = init_beta
        self._beta = self._init_beta
        self.min_beta = min_beta
        self.decay_beta_amount = decay_beta_amount

        self.seed = seed
        self.rng_sample = utils.initialize_random_generator(seed)

        self.create_abstraction_directory()
        self.initialize_abstraction()
        self.i = 0
        self.max_size = 100000 # number of concrete states to keep in memory for an abstract state
        self.refinement_threshold = None

    def create_abstraction_directory(self):
        directory = f"{self._directory}/{self.abstraction_directory}/"
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def initialize_abstraction(self):
        '''
        Initializes _tree, init_abs_action_list, and abs_to_con
        '''
        self._tree: CAT = CAT(state_ranges = self._env.state_ranges, 
                is_int_state_variable = self._env.is_int_state_variable, 
                allowed_diff_to_refine = self._allowed_diff_to_refine, 
                flexible_refinement = self._flexible_refinement)
        if self.reuse_cat_path is not None:
            self._tree.load_cat(self.reuse_cat_path)
        if self._plot_abstractions:
            self.visualizer = VisualizeAbstraction()
            self.visualizer.initialize_decision_boundaries()

        # create abstract states and actions
        for _ in range(self._init_state_abs_level):
            for refine_vars in self._env.initial_refinement_order:
                unstable_states = list(self._tree._leaves.keys())
                for unstable_state in unstable_states:
                    self._tree.refine_state_uniform_partitioning(unstable_state, refine_variables=refine_vars)
                    if self._plot_abstractions:
                        self.visualizer.update_decision_boundaries(clf=None, parent_id=unstable_state.id, last_used_id=None, uniform_partitioning=True)
                        self.visualizer.plot_decision_boundaries("inital_abstraction")
            
        self._tree.plot_cat(self._directory, 1, {})
        print("Initialized state abstraction and plotted CAT")
        self.init_abs_action_list = self.initialize_action_abstraction()
        self._tree.update_n_abstract_states()
        self.update_n_abstract_actions()

        self.initialize_empty_abs_to_con()

    def initialize_action_abstraction(self):
        # create abstract actions 
        init_abs_action_list = []
        if self._is_action_space_discrete:
            for discrete_action in range(self._env._action_space.n):
                init_abs_action_list.append(AbstractAction([], discrete_action))
        else:
            for discrete_action, parameter_ranges in self._env.action_param_ranges.items():
                init_abs_action_list.append(AbstractAction(parameter_ranges, discrete_action))

        if not self._is_action_space_discrete:
            for _ in range(self._init_action_abs_level):
                new_abstract_actions = []
                for abs_action in init_abs_action_list:
                    new_action_ranges = self._tree.refine_action(abs_action.action_params, self._env.is_int_discrete_action_params[abs_action.discrete_action])
                    new_actions = [AbstractAction(action_range, abs_action.discrete_action) for action_range in new_action_ranges]
                    new_abstract_actions.extend(new_actions)
                init_abs_action_list = copy.deepcopy(new_abstract_actions)
        print("Initialized action abstraction:", init_abs_action_list)
        return init_abs_action_list
    
    def update_n_abstract_actions(self):
        self._n_abstract_actions = 0
        for leaf in self._tree._leaves:
            if leaf in self._agent._qtable._qtable:
                self._n_abstract_actions += len(self._agent._qtable._qtable[leaf])
            else:
                self._n_abstract_actions += len(self.init_abs_action_list)


    def update_abstraction(self, epi_i, tderror_buffer, tderror_buffer_s_absa, qvalue_buffer, qvalue_buffer_s_absa, fraction=1.0):
        print(f"Fraction: {fraction}")
        tderror_buffer = self.clean_buffer(deepcopy(tderror_buffer._buffer))
        qvalue_buffer = self.clean_buffer(deepcopy(qvalue_buffer._buffer))
        tderror_buffer_s_absa = tderror_buffer_s_absa._buffer
        qvalue_buffer_s_absa = qvalue_buffer_s_absa._buffer
        dispersion_log_state = {}
        dispersion_log_action = {}
        unstable_states = set()
        unstable_actions = set()
        if len(tderror_buffer)>0:
            # Selection of abstract states and actions
            unstable_states, unstable_actions, dispersion_log_state, dispersion_log_action = self.find_k_unstable_states_actions(tderror_buffer, qvalue_buffer, qvalue_buffer_s_absa, fraction)

            # Refinement of abstract states and actions
            old_to_new = {}
            for unstable_state_action in set(unstable_states):
                new_abstract_states = self.refine_abstract_state(unstable_state_action[0], unstable_state_action[1], tderror_buffer_s_absa, qvalue_buffer_s_absa)
                old_to_new[unstable_state_action[0]] = new_abstract_states

            if not self._is_action_space_discrete:
                for unstable_state_action in unstable_actions:
                    unstable_state, unstable_action = unstable_state_action[0], unstable_state_action[1]
                    if unstable_state in old_to_new:
                        new_abstract_states = old_to_new[unstable_state]
                    else:
                        new_abstract_states = [unstable_state]
                    self.refine_abstract_action(unstable_state, new_abstract_states, unstable_action)

            print(f'Number of states refined: {len(unstable_states)}')

        self.write_action_space(epi_i, dispersion_log_state, dispersion_log_action)
        return len(unstable_states), len(unstable_actions)

    def write_action_space(self, epi_i, dispersion_log_state, dispersion_log_action):
        self.create_abstraction_directory()
        sorted_dict = {}

        for state_action, value in dispersion_log_state.items():
            sorted_dict["State:"+str(state_action)] = value

        for state_action, value in dispersion_log_action.items():
            sorted_dict["Action:"+str(state_action)] = value

        abstract_action_space = {}
        for state in self._agent._qtable._qtable:
            for action in self._agent._qtable._qtable[state]:
                if str(state) not in abstract_action_space:
                    abstract_action_space[str(state)] = {}
                if str(action) not in abstract_action_space[str(state)]:
                    abstract_action_space[str(state)][str(action)] = 1
        abstract_action_space = dict(sorted(abstract_action_space.items(), key=lambda item: len(item[1]), reverse=True))
        for state in abstract_action_space:
            sorted_dict[state] = abstract_action_space[state]

        directory = f"{self._directory}/{self.abstraction_directory}/action_space"
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(f"{directory}/action_space_{epi_i}.txt", 'w') as file:
            json.dump(sorted_dict, file, indent=1)

    def refine_abstract_state(self, unstable_state, unstable_action, tderror_buffer_s_absa, qvalue_buffer_s_absa):
        new_abstract_states = []
        print(f"Refining state: {unstable_state}")
        last_used_id = self._tree._n_abstract_nodes-1
        if not self._flexible_refinement:
            # find unstable variables and refine the state with uniform partitioning
            if self._refinement_method == "aggressive":
                vector = [1]*(len(unstable_state)//2)
            elif self._refinement_method == "deliberative":
                vector = self.find_unstable_variables(unstable_state, self._maximum_state_variables_to_split)
            if 1 not in vector:
                return []
            new_abstract_states = self._tree.refine_state_uniform_partitioning(unstable_state, vector)
            if self._plot_abstractions:
                self.visualizer.update_decision_boundaries(None, unstable_state.id, last_used_id, uniform_partitioning=True)
        else:
            # refine the state with flexible partitioning
            new_abstract_states = []
            clf, mask, min_vals, max_vals, class_to_states, class_to_minm_maxm = self.find_flexible_refinement(unstable_state, unstable_action, tderror_buffer_s_absa, qvalue_buffer_s_absa)
            if clf is not None:
                new_abstract_states = self._tree.refine_state_flexible_refinement(unstable_state, clf, mask, min_vals, max_vals, class_to_states, class_to_minm_maxm)
                if self._plot_abstractions:
                    self.visualizer.update_decision_boundaries(clf, unstable_state.id, last_used_id)
            else:
                new_abstract_states.append(unstable_state)

        # initializing q-values of new abstract states to q-values of their parents
        if unstable_state in self._agent._qtable._qtable:
            actions_to_qvalues = copy.deepcopy(self._agent._qtable._qtable[unstable_state])
            for new_state_abs in new_abstract_states:
                self._agent.initialize_qvalues(new_state_abs, actions_to_qvalues, self.init_abs_action_list)
                # self._agent._tderror_buffer._table[new_state_abs] = {}
        else:
            print("Unstable state not found in qtable")

        if unstable_state not in new_abstract_states:
            self._agent.delete_state_from_tderror_buffer(unstable_state)
            self._agent.delete_state_from_qtable(unstable_state)
            
        self._tree.update_n_abstract_states()
        print(f"\tNew states: {new_abstract_states}")
        return new_abstract_states

    def refine_abstract_action(self, unstable_state, new_abstract_states, unstable_action):  
        if self._env.is_int_discrete_action_params[unstable_action.discrete_action] is None:
            return []
        print(f"For state: {unstable_state}, refining action: {unstable_action}")
        new_action_ranges = self._tree.refine_action(unstable_action.action_params, self._env.is_int_discrete_action_params[unstable_action.discrete_action])
        new_actions = [AbstractAction(action_range, unstable_action.discrete_action) for action_range in new_action_ranges]
        for abstract_state in new_abstract_states:
            if unstable_action in new_actions:
                pass
            elif unstable_action in self._agent._qtable._qtable[abstract_state]:
                for new_action in new_actions:
                    self._agent._qtable._qtable[abstract_state][new_action] = self._agent._qtable._qtable[abstract_state][unstable_action]
                del self._agent._qtable._qtable[abstract_state][unstable_action]
            else:
                for new_action in new_actions:
                    self._agent._qtable._qtable[abstract_state][new_action] = self._agent._initial_value
            if unstable_state in self._agent._qtable._qtable:
                for action in self._agent._qtable._qtable[unstable_state].keys():
                    if action not in self._agent._qtable._qtable[abstract_state]:
                        self._agent._qtable._qtable[abstract_state][action] = self._agent._initial_value
        print(f"\tNew actions: {new_actions}")
        self.update_n_abstract_actions()
        return new_actions


    def state(self, state):
        try:
            node = self._tree.state_recursive_general(state, start_node=self._tree._root)
            assert node is not None, f'abstract state for {state} not found in tree, node is None'
            return node
        except Exception as e:
            print(self._env.state_ranges)
            print([item for item in state])
            print(f'abstract state for {state} not found in tree')
            for i in range(len(state)-1):
                if state[i] >= self._tree._root_name[2*i] and state[i] < self._tree._root_name[2*i+1]:
                    print(True)
                else:
                    print(False)
            raise e

    # to get an abtract action for a grounded action
    def action(self, state, action):
        abstract_actions = self._agent._qtable.get_actions(state)
        discrete_action = int(action[0])
        action_params = action[1]
        maximum = []     
        for paramter_range in self._env.action_param_ranges[discrete_action]:
            maximum.append(paramter_range[1])      
        for abs_action in abstract_actions:
            assert abs_action is not None, f"abs_action :{abs_action} discrete_action: {discrete_action}"
            if abs_action.discrete_action == discrete_action:
                abs_action_temp = []
                for item in abs_action.action_params:
                    abs_action_temp.extend(item)
                if self._tree.fallsWithinNew(action_params, abs_action_temp, [i for i in range(len(action_params))], maximum):
                    return abs_action
        return None
    
    # to sample grounded action from abstract action
    def sample_action(self, abstract_action, use_mean_action=False):
        discrete_action = abstract_action.discrete_action
        params = abstract_action.action_params
        if use_mean_action:
            action_params = [np.mean([params[i][0], params[i][1]]) for i in range(len(params))]
        else:
            action_params = [self.rng_sample.uniform(params[i][0], params[i][1]) for i in range(len(params))]
        return (discrete_action, action_params)

    def add_concrete_state(self, abstract_state, concrete_state):
        self.initialize_concrete_states(abstract_state)
        if len(self.abs_to_con[abstract_state]) >= self.max_size:
            self.abs_to_con[abstract_state].pop()
        self.abs_to_con[abstract_state].add(concrete_state)

    def add_concrete_states(self, abstract_state, concrete_states):
        for cs in concrete_states:
            self.add_concrete_state(abstract_state, cs)

    def initialize_concrete_states(self, abstract_state):
        if abstract_state not in self.abs_to_con:
            self.abs_to_con[abstract_state] = set()

    def initialize_empty_abs_to_con(self):
        self.abs_to_con = {}

    ########################### Find unstable states ############################
    def clean_buffer(self, buffer_in):
        buffer = deepcopy(buffer_in)
        indivisible_states = []
        for state in buffer:
            valid = False
            node = self._tree.find_node(state)
            if node._parent is None: 
                split = self._env.state_ranges
                for i in range (len(state.state)//2):
                    lower = split [i][0]
                    upper = split [i][1]
                    if self._tree.is_refinable([lower, upper], self._env.is_int_state_variable[i]): 
                        valid = True
                        break
                if not valid: 
                    indivisible_states.append(state)
            else: 
                for i in range (len(state.state)//2):
                    lower = state.state[2*i]
                    upper = state.state[2*i+1]
                    if self._tree.is_refinable([lower, upper], self._env.is_int_state_variable[i]): 
                        valid = True
                        break
                if not valid: 
                    indivisible_states.append(state)

        for s in indivisible_states:
            del buffer[s]
        return buffer
    
    def get_minm_maxm_tderror(self, buffer):
        all_values = []
        for state in buffer:
            for action in buffer[state]:
                all_values.extend(buffer[state][action])
        if len(all_values) > 0:
            minm = min(all_values)
            maxm = max(all_values)
            return minm, maxm
        return None, None
    
    def normalize_eval(self, in_buffer, minm, maxm):
        buffer = deepcopy(in_buffer)
        for state in in_buffer:
            for action in in_buffer[state]:
                buffer[state][action] = []
                for eval_value in in_buffer[state][action]: 
                    buffer[state][action].append((eval_value - minm) / (maxm - minm))
        return buffer

        
    ################ beta schedular ############################
    def decay_beta(self): 
        if self._beta > self.min_beta:
            self._beta -= self.decay_beta_amount

    def reset_beta(self):
        self._beta = self._init_beta


    ################################## Select Abstract States and Abstract Actions for refinement ######################################
    def find_k_unstable_states_actions(self, tderror_buffer, qvalue_buffer, qvalue_buffer_s_absa, fraction):
        """
            tderror_buffer: dict[state][action] -> sequence/array of TD-errors
        """
            
        ################################## State selection ######################################
        state_action_to_var = {}                              # variation for every (state, action) pair
        state_most_unstable_action_to_var = {}                # for each state, only the most “unstable” action (max variation)
        state_most_unstable_action_to_var_concrete = {}       # for each state, only the most “unstable” action (max variation) by looking at concrete states

        buffer = deepcopy(tderror_buffer)
        for state in buffer:
            action_to_variation = {}

            for abs_action in buffer[state]:
                if len(buffer[state][abs_action]) > 0:
                    mean = np.mean(buffer[state][abs_action])
                    mean = max(mean, 1.0)  
                    variation = np.std(buffer[state][abs_action]) / mean
                    if np.isnan(variation): 
                        variation = 0
                    action_to_variation[abs_action] = variation

            if len(action_to_variation) > 0:
                for action, var in action_to_variation.items():
                    state_action_to_var[(state, action)] = var

                max_var_action = max(action_to_variation, key=action_to_variation.get)
                max_var = action_to_variation[max_var_action]
                state_most_unstable_action_to_var[(state, max_var_action)] = max_var

        buffer_s = deepcopy(qvalue_buffer_s_absa)
        for abs_state, abs_actions in buffer.items():
            concrete_states = list(self.abs_to_con[abs_state])
            if len(concrete_states) <= 1 or len(abs_actions) == 0:
                continue

            action_to_variation = {}
            for abs_action in abs_actions:
                # collect mean Q-values across all concrete states for this action
                q_means = [
                    float(np.mean(np.atleast_1d(buffer_s[state][abs_action])))
                    for state in concrete_states
                    if state in buffer_s and abs_action in buffer_s[state] and len(buffer_s[state][abs_action]) > 0
                ]
                if len(q_means) > 1:
                    q_means_arr = np.asarray(q_means, dtype=float)
                    variation = float(np.std(q_means_arr) / np.mean(q_means_arr))
                    action_to_variation[abs_action] = variation

            if len(action_to_variation) > 0:
                max_var_action = max(action_to_variation, key=action_to_variation.get)
                max_var = action_to_variation[max_var_action]
                state_most_unstable_action_to_var_concrete[(abs_state, max_var_action)] = max_var

        state_action_to_var = dict(sorted(state_action_to_var.items(), key=lambda item: item[1], reverse=True))
        state_most_unstable_action_to_var = dict(sorted(state_most_unstable_action_to_var.items(), key=lambda item: item[1], reverse=True))
        state_most_unstable_action_to_var_concrete = dict(sorted(state_most_unstable_action_to_var_concrete.items(), key=lambda item: item[1], reverse=True))
        # print(f"Total number of states in buffer: {len(state_most_unstable_action_to_var_concrete)}")

        combined_dict = {}
        for (state, action), var in state_most_unstable_action_to_var.items():
            if state not in combined_dict:
                combined_dict[state] = {}
            if action not in combined_dict[state]:
                combined_dict[state][action] = [0.0, 0.0]
            combined_dict[state][action][0] = var
        for (state, action), var in state_most_unstable_action_to_var_concrete.items():
            if state not in combined_dict:
                combined_dict[state] = {}
            if action not in combined_dict[state]:
                combined_dict[state][action] = [0.0, 0.0]
            combined_dict[state][action][1] = var

        heterogenity_values = {}
        for abs_state in combined_dict.keys():
            for abs_action in combined_dict[abs_state].keys():
                heterogenity_values[(abs_state, abs_action)] = self._beta * combined_dict[abs_state][abs_action][0] + (1.0 - self._beta) * combined_dict[abs_state][abs_action][1]

        heterogenity_values = dict(sorted(heterogenity_values.items(), key=lambda item: item[1], reverse=True))

        # pairs of (unstable_state, unstable_action): focus is on state so will contain a single action for a state
        var_values = [x for x in list(heterogenity_values.values()) if round(x,10) > 0.0]
        count = len(var_values)
        k = int(fraction * count)
        k = min(k, self._k_cap)
        print(f"fraction: {fraction}, k: {k}")

        unstable_states = list(heterogenity_values.keys())[:k]
        unstable_states = sorted(unstable_states, key=lambda s: s[0].id)
        print(f"Selected unstable states: {unstable_states}")

        dispersion_log_state = {}
        for state in  list(state_most_unstable_action_to_var.keys()):
            if state in unstable_states:
                dispersion_log_state[state] = str(round(state_most_unstable_action_to_var[state],5))
            else:
                dispersion_log_state[state] = str(round(state_most_unstable_action_to_var[state],5))+" (not selected)" 

        ################################## Action selection ######################################
        state_action_to_var_ = {}
        state_most_unstable_action_to_var_ = {}

        for state in buffer:
            action_to_variation = {}

            for abs_action in buffer[state]:
                if len(buffer[state][abs_action]) > 0:
                    mean = np.mean(buffer[state][abs_action])
                    mean = max(mean, 1.0)
                    variation = np.std(buffer[state][abs_action]) / mean
                    if np.isnan(variation): 
                        variation = 0
                    action_to_variation[abs_action] = variation

            if len(action_to_variation) > 0:
                for action, var in action_to_variation.items():
                    state_action_to_var_[(state, action)] = var
                max_var_action = max(action_to_variation, key=action_to_variation.get)
                max_var = action_to_variation[max_var_action]
                state_most_unstable_action_to_var_[(state, max_var_action)] = max_var

        state_action_to_var_ = dict(sorted(state_action_to_var_.items(), key=lambda item: item[1], reverse=True))
        state_most_unstable_action_to_var_ = dict(sorted(state_most_unstable_action_to_var_.items(), key=lambda item: item[1], reverse=True))

        # pairs of (unstable_state, unstable_action): focus is on action so may contain multiple actions for a state 
        var_values = [x for x in list(state_action_to_var_.values()) if x > 0.0]
        threshold_var = np.mean(var_values) if len(var_values) > 0 else 0
        count = len([1 if x > threshold_var else 0 for x in var_values])
        k = int(fraction * count)
        k = min(k, self._k_cap_actions)

        unstable_selected = list(state_action_to_var_.keys())
        unstable_selected = unstable_selected[0:k]
        unstable_actions = [item for item in unstable_selected if round(state_action_to_var_[item],10) > 0.0]

        dispersion_log_action = {}
        for action in list(state_action_to_var_.keys()):
            if action in unstable_actions:
                dispersion_log_action[action] = str(round(state_action_to_var_[action],5))
            else:
                dispersion_log_action[action] = str(round(state_action_to_var_[action],5))+" (not selected)"

        return unstable_states, unstable_actions, dispersion_log_state, dispersion_log_action

    @ignore_warnings(category=ConvergenceWarning)
    def get_total_unstable_number(self, variation_values, n_clusters=3, top_two=False):
        # print(variation_values)
        if len(variation_values) > 0:
            X = []
            for i in range (len(variation_values)):
                item = variation_values[i]
                X.append([item])
            X = np.array(X)
            X.reshape(-1, 1)
            model = AgglomerativeClustering(n_clusters=min(len(variation_values),n_clusters)).fit(X)
            res = model.labels_
            ref1 = res[0]
            ref2 = None
            num = 0
            for i in range(len(res)):
                if ref2 == None and res[i] != ref1:
                    ref2 = res[i]
                if top_two:
                    if res[i] == ref1 or res[i] == ref2: 
                        num += 1
                else:
                    if res[i] == ref1: 
                        num += 1          
            return num
        else: 
            return 0

    ########################### Refine unstable states ############################
    def split_abs_state_wrs (self, abs_state, wrt_variable_index):
        abs_state_1 = list(abs_state)
        abs_state_2 = list(abs_state)
        state_value = abs_state[wrt_variable_index]
        
        interval = state_value.split(",")
        for i in range(2): interval[i] = int(interval[i])
        midpoint = int((interval[1] - interval[0])/2) + interval[0] 
        interval1 = str(interval[0]) + "," + str(midpoint)
        interval2 = str(midpoint) + "," + str(interval[1])  
        abs_state_1[wrt_variable_index] = interval1
        abs_state_2[wrt_variable_index] = interval2
        return [(*abs_state_1, ), (*abs_state_2, )]

    @ignore_warnings(category=ConvergenceWarning)
    def find_unstable_variables(self, unstable_state, max_variables_to_split):
        vars = self.choose_vars_1(unstable_state)
        vector = [0]*len(self._env.state_ranges)
        if len(self._env.state_ranges) <= max_variables_to_split:
            vector = list((np.array(vars) > 0).astype(int) )
        else: 
            for var, index in sorted(zip(vars, list(range(len(vars)))), reverse=True)[:max_variables_to_split]:
                if var!=0:
                    vector[index] = 1
        return vector

    def choose_vars_1(self, unstable_state):
        vars = []
        variance = self.get_variation(unstable_state)
        for k in range(0,len(unstable_state.state),2):
            if self._tree.is_refinable([unstable_state.state[k],unstable_state.state[k+1]], self._env.is_int_state_variable[k//2]):
                vars.append(variance[int(k/2)])
            else: 
                vars.append(0) 
        return vars

    def get_variation(self, abs_state):
        if abs_state in self._agent._qtable._qtable:
            if len(self.abs_to_con[abs_state]) > 0:
                return np.std(np.array(list(self.abs_to_con[abs_state])), axis = 0)
            else:
                return np.zeros(len(abs_state.state)//2)
        else:
            return np.zeros(len(abs_state.state)//2)

    def find_flexible_refinement(self, unstable_state, unstable_action, tderror_buffer_s_absa, qvalue_buffer_s_absa):        
        state_to_value = dict()
        lookup_table = tderror_buffer_s_absa
        if len(self.abs_to_con[unstable_state]) > 1:
            for c in self.abs_to_con[unstable_state]:
                if c in lookup_table:
                    if unstable_action in lookup_table[c]:
                        state_to_value[c] = np.mean(lookup_table[c][unstable_action])
        if len(state_to_value) == 0:
            print(f"No concrete states found for refinement {self.abs_to_con[unstable_state]}")
            return None, {}, {}, {}, {}, {}
        
        state_to_value = OrderedDict(sorted(state_to_value.items(), key=lambda t: t[1], reverse=True))     

        # new prepare states and values 
        states_np = np.array(list(state_to_value.keys()))
        states_np_mask = np.ptp(states_np, axis=0) != 0
        states_np_reduced  = states_np[:, states_np_mask]

        values = np.array(list(state_to_value.values()))

        states_min = states_np_reduced.min(axis=0)
        states_max = states_np_reduced.max(axis=0)
        states_range = states_max - states_min
        states_norm = (states_np_reduced - states_min) / states_range

        values_min = values.min()
        values_max = values.max()
        values_range = values_max - values_min
        values_norm = (values - values_min) / values_range

        X = states_norm
        unstable_values = values_norm

        # find clusters and decision boundaries
        print(f"\tUnstable state: {unstable_state.state}")
        directory = f"{self._directory}/{self.abstraction_directory}/clusters"
        filename = f"{directory}/clusters_abs_{len(self._tree._leaves)}_{str(unstable_state)}.png"
        self.create_abstraction_directory()
        if not os.path.exists(directory):
            os.makedirs(directory)

        flexible_refinement = FlexibleRefinement(unstable_state, X, unstable_values, self._min_samples, self._max_clusters, self.kernel, self._plot_abstractions, filename)

        clf = None
        class_to_states = {}
        class_to_minm_maxm = {}

        if not np.isnan(unstable_values).any() and  len(unstable_values) >= self._max_clusters:
            n_clusters, labels, clf, y_pred = flexible_refinement.find_clusters()
        
            if clf is not None:
                for decision in labels:
                    class_to_states[decision] = set()
                for i in range(len(labels)):
                    if y_pred[i] not in class_to_states:
                        class_to_states[y_pred[i]] = set()
                    class_to_states[y_pred[i]].add(tuple(states_np[i,:]))

                for class_ in class_to_states:
                    class_to_minm_maxm[class_] = self.get_minm_maxm(class_to_states[class_])
        
        self.i += 1
        return clf, states_np_mask, states_min, states_max, class_to_states, class_to_minm_maxm

    def get_minm_maxm(self, concrete_states):
        minm_maxm = []
        if len(concrete_states) == 0:
            return minm_maxm
        transposed_matrix = list(zip(*concrete_states))
        # Find min and max along each axis (column)
        min_values = [min(column) for column in transposed_matrix]
        max_values = [max(column) for column in transposed_matrix]
        for i in range(len(min_values)):
            minm_maxm.append((min_values[i],max_values[i]))
        return minm_maxm

    def compute_weight(self, decisions):
        weight = {}
        for decision in decisions:
            if decision not in weight:
                weight[decision] = 1
            else:
                weight[decision] += 1
        for decision in weight:
            weight[decision] = 1000/weight[decision]
        return weight

 