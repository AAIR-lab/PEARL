import math
import copy 

from src.data_structures.qvalue_table import Qtable
from src.data_structures.e_table import Etable
from src.data_structures.buffer import TDErrorBuffer, QvalueBuffer, TraceBuffer
from src.misc import utils



class AbstractTDlambdaAgent:
    def __init__(self, seed, is_action_space_discrete, action_size,  gamma, alpha, eps_min, decay, _lambda, epsilon = 1):
        self._epsilon = epsilon
        self._epsilon_min = eps_min
        self._gamma = gamma
        self._alpha = alpha
        self._initial_value = 0
        self._decay = decay
        self._lambda = _lambda
        self.is_action_space_discrete = is_action_space_discrete
        self.action_size = action_size
        self.abs_state_to_abs_actions = {} 

        self._qtable = Qtable()
        self._etable = Etable()
        self._qtable_s_absa = Qtable()

        self.initialize_buffers()
        
        self.rng_eval = utils.initialize_random_generator(seed)
        self.rng = utils.initialize_random_generator(seed)
        self.rng_dispersion = utils.initialize_random_generator(seed)

        self.min_qvalue = math.inf
        self.max_qvalue = -math.inf


    ###################### Compute Q-values for abstract states and abstract actions ################################### 
    def initialize_qvalues(self, state_abs, actions_to_qvalues=None, init_abs_action_list=None): 
        if state_abs not in self._qtable._qtable:
            if actions_to_qvalues is None:
                for abs_action in init_abs_action_list:
                    self._qtable.initialize_qvalue(state_abs, abs_action, self._initial_value)
                self.min_qvalue = min(self.min_qvalue, self._initial_value)
                self.max_qvalue = max(self.max_qvalue, self._initial_value)
            else:
                for abs_action, qvalue in actions_to_qvalues.items():
                    self._qtable.update_qvalue(state_abs, abs_action, qvalue)
    
    def clear_etable(self):
        self._etable = Etable()

    def update_qvalue(self, sample):  
        state, action, next_state, reward, done = sample.state_abs, sample.action_abs, sample.next_state_abs, sample.reward_abs, sample.done

        # compute TD error
        old_state_qvalue = self._qtable.get_qvalue(state, action, self._initial_value)

        if done:
            td_target = reward
        else:
            td_target = reward + self._gamma * self._qtable.get_max_qvalue(next_state, self._initial_value)

        td_error = td_target - old_state_qvalue

        # Initialize eligibilities for all actions in the current state to 0 (sets only if not in the e-table)
        actions = self._qtable.get_actions(state)
        for action_ in actions:
            self._etable.initialize_value(state, action_)

        # Increment eligibility of the current (state, action)
        current_eligibility = self._etable.get_value(state, action)
        self._etable.update_value(state, action, current_eligibility + 1)

        # Update all Q-values using current eligibility traces
        for s in self._etable._etable:
            for a in self._etable._etable[s]:
                eligibility = self._etable.get_value(s, a)
                if eligibility == 0.0:
                    continue
                self._qtable.update_qvalue(s, a, self._qtable.get_qvalue(s, a, self._initial_value) + self._alpha * td_error * eligibility)

        # Decay all eligibility traces
        for s in self._etable._etable:
            for a in self._etable._etable[s]:
                eligibility = self._etable.get_value(s, a)
                self._etable.update_value(s, a, eligibility * self._gamma * self._lambda)

        new_state_qvalue = self._qtable.get_qvalue(state, action, self._initial_value)
        self.min_qvalue = min(self.min_qvalue, new_state_qvalue)
        self.max_qvalue = max(self.max_qvalue, new_state_qvalue)
    
    def get_random_action(self, rng, state_abs, init_abs_action_list):
        action_index = rng.randint(0, self.action_size-1)  
        all_actions = self._qtable.get_actions(state_abs)
        actions = [action for action in all_actions if action.discrete_action == action_index]
        if len(actions) == 0:
            actions = init_abs_action_list
        action = rng.choice(actions) 
        return action

    def policy(self, state_abs, init_abs_action_list):
        if self.rng.uniform(0,1) < self._epsilon: 
            action = self.get_random_action(self.rng, state_abs, init_abs_action_list)
        else:
            action = self._qtable.get_best_action(state_abs, self.rng)
            if action is None:
                action = self.get_random_action(self.rng, state_abs, init_abs_action_list)
        return action

    def evaluation_policy(self, state_abs, init_abs_action_list):
        action = self._qtable.get_best_action(state_abs, self.rng_eval)
        if action is None:
            action = self.get_random_action(self.rng_eval, state_abs, init_abs_action_list)
        return action

    def decay_epsilon(self):
        if self._epsilon > self._epsilon_min: 
            self._epsilon =  self._epsilon * self._decay
        self._epsilon = max(self._epsilon, self._epsilon_min)


    ######################## Estimate Q-values for concrete states and abstract actions ##############################
    def estimate_concrete_qvalue(self, sample):
        state, action, next_state_abs, reward, done = sample.rounded_state, sample.action, sample.next_state_abs, sample.reward, sample.done
        state_abs, action_abs, next_state_abs, reward_abs = sample.state_abs, sample.action_abs, sample.next_state_abs, sample.reward_abs
        
        # compute TD error
        old_qvalue = self._qtable_s_absa.get_qvalue(state, action_abs, self._initial_value)

        if done:
            td_target = reward
        else:
            td_target = reward + self._gamma * self._qtable.get_max_qvalue(next_state_abs, self._initial_value)

        td_error = td_target - old_qvalue

        # update Q-value
        new_state_qvalue = old_qvalue + self._alpha * td_error
        self._qtable_s_absa.update_qvalue(state, action_abs, new_state_qvalue)

    ################################################################################################################################
    ############################# Initialize Buffers ##############################
    def freeze_qtable_for_tderror_computation(self):
        self._frozen_qtable = Qtable()
        self._frozen_qtable._qtable = copy.deepcopy(self._qtable._qtable)
        self._frozen_qtable_s_absa = Qtable()
        self._frozen_qtable_s_absa._qtable = copy.deepcopy(self._qtable_s_absa._qtable)

    def initialize_buffers(self):
        self._tderror_buffer = TDErrorBuffer()
        self._qvalue_buffer = QvalueBuffer()
        self._tderror_buffer_s_absa = TDErrorBuffer()
        self._qvalue_buffer_s_absa = QvalueBuffer()
        self.freeze_qtable_for_tderror_computation()
            

    ############## TD error and Qvalue Buffers for abstract states and abstract actions ############               
    def add_measure_to_buffer(self, sample):
        state, next_state, action, reward, done = sample.state_abs, sample.next_state_abs, sample.action_abs, sample.reward_abs, sample.done
        
        old_qvalue = self._frozen_qtable.get_qvalue(state, action, self._initial_value)

        if done:
            td_target = reward
        else:
            td_target = reward + self._gamma * self._frozen_qtable.get_max_qvalue(next_state, self._initial_value)

        td_error = td_target - old_qvalue

        self._tderror_buffer.add(state, action, td_error)
        self._qvalue_buffer.add(state, action, old_qvalue + self._alpha * td_error)


    ############### TD error Buffer for concrete states and abstract actions ################
    def add_measure_to_concrete_tderror(self, sample):
        state, action, next_state_abs, reward, done = sample.rounded_state, sample.action, sample.next_state_abs, sample.reward, sample.done
        state_abs, action_abs, next_state_abs, reward_abs = sample.state_abs, sample.action_abs, sample.next_state_abs, sample.reward_abs
        
        old_qvalue = self._frozen_qtable_s_absa.get_qvalue(state, action_abs, self._initial_value)

        if done:
            target = reward + self._gamma * self._initial_value
        else:
            target = reward + self._gamma * self._frozen_qtable.get_max_qvalue(next_state_abs, self._initial_value)

        td_error = target - old_qvalue

        self._tderror_buffer_s_absa.add(state, action_abs, td_error)
        self._qvalue_buffer_s_absa.add(state, action_abs, old_qvalue + self._alpha * td_error)


    def delete_state_from_tderror_buffer(self, state):
        if state in self._tderror_buffer._buffer:
            del self._tderror_buffer._buffer[state]

    def delete_state_from_qtable(self, state):
        if state in self._qtable._qtable:
            del self._qtable._qtable[state]
