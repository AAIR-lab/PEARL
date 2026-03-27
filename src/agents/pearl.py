import numpy as np
import copy
import time
import sympy as sp
np.set_printoptions(suppress=True)
# np.set_printoptions(legacy='1.25')

from src.data_structures.trace import Transition, Trace

class PEARL:

    def __init__(self, seed, env, agent, agent_con, abstract, log, episode_max, step_max, abs_interval, eval_episodes, directory):
        self.seed = seed
        self.env = env
        self.agent = agent
        self.agent_con = agent_con
        self.abstract = abstract
        self.log = log
        self.episode_max = episode_max
        self.step_max = step_max
        self.abs_interval = abs_interval
        self.eval_episodes = eval_episodes
        self.directory = directory
        self.a = None

    def main(self):
        epi_i = 0
        self.evaluation_mean_succ = 0
        self.n_unstable_states, self.n_unstable_actions = 0, 0
        self.start_time = time.time()

        while (epi_i <= self.episode_max):
            epi_i += 1

            ################### learn for an episode ###################
            success, reward, epoch, final_state, _, _ = self.abstract_qlearning_episode(epi_i, self.env, self.agent, self.abstract, self.abstract, evaluating=False)
            self.abstract._tree.update_n_abstract_states()
            print("learning epi: " + str(epi_i) + '  ' + 
                    "epsilon: " + str(round(self.agent._epsilon,3)) + '  ' + 
                    "reward: " + str (round(reward,2)) + '  ' + 
                    "success: " + str(success) + '  ' + 
                    "steps: " + str(epoch) + '  ' + 
                    "#states: " + str(self.abstract._tree._n_abstract_states) + '  ' + 
                    "#actions: " + str(self.abstract._n_abstract_actions) + '  ' + '\n\t' +
                    "meanreward: " + str (round (self.log.recent_mean_learning_reward(self.abs_interval),2)) + '  ' +
                    "meansuccess: " + str (round (self.log.recent_mean_learning_success(self.abs_interval),2)) + '  ' +
                    "meansteps: " + str (round (self.log.recent_mean_learning_steps(self.abs_interval),2)) + '  ' +
                    "finalstate: " + str([round(item,2) for item in final_state])
                    )

            ################### evaluate and refine abstraction ###################
            if epi_i > 0 and int(epi_i % self.abs_interval) == 0:
                if self.abstract._plot_abstractions:
                    self.abstract.visualizer.plot_decision_boundaries(filename=f"img_before_refinement_epi_{epi_i}_abs_{len(self.abstract._tree._leaves)}")
            
                ################### evaluate policy ###################
                self.evaluation_mean_succ = self.evaluate_policy(epi_i, self.env, self.agent, self.abstract, n_epi=self.eval_episodes)
                # if self.evaluation_mean_succ >= 0.9:
                #     break

                ################### refine abstraction ###################
                print("Updating abstraction...")
                fraction = self.refinement_fraction(epi_i, self.agent._epsilon, self.abstract._beta, self.evaluation_mean_succ)
                self.n_unstable_states, self.n_unstable_actions = self.abstract.update_abstraction(epi_i, self.agent._tderror_buffer, self.agent._tderror_buffer_s_absa, self.agent._qvalue_buffer, self.agent._qvalue_buffer_s_absa, fraction=fraction)
                self.abstract.decay_beta()
                if self.abstract._plot_abstractions:
                    self.abstract.visualizer.plot_decision_boundaries(filename=f"img_after_refinement_epi_{epi_i}_abs_{len(self.abstract._tree._leaves)}")
                self.abstract._tree.plot_cat(self.directory, 1, {}) 
                self.log.log_td(epi_i, self.agent._tderror_buffer._buffer, self.agent._qtable._qtable)
                self.log.plot_result(100, self.directory+"/result")           
                print('Updated CAT!')
                self.agent.initialize_buffers() # Empty buffers and Fix Qtable for tderror computation
                self.log.save_execution(self.directory, "results")
                self.log.print_qtable(self.agent._qtable, f"{self.directory}/qtable", epi_i)
          
            self.log._summary_writer.add_scalar("time", time.time() - self.start_time, epi_i)

        self.log.close_writer()
        self.log.save_execution(self.directory, "results")
        self.log.plot_result(100, self.directory+"/result") 
        return self.log.recent_mean_learning_reward(self.abs_interval)

    @staticmethod
    def find_a_d_symbolic(x_values, y_values):
        if len(x_values) != 2 or len(y_values) != 2:
            raise ValueError("Exactly two points are required for symbolic solving.")
        x1, x2 = x_values
        y1, y2 = y_values
        a, d = sp.symbols('a d')
        eq1 = sp.Eq((a / (x1 + 1)) + d, y1)
        eq2 = sp.Eq((a / (x2 + 1)) + d, y2)
        solution = sp.solve((eq1, eq2), (a, d))
        return float(solution[a]), float(solution[d])

    def refinement_fraction(self, epi_i, epsilon, beta, evaluation_success):
        if evaluation_success >= 0.9:
            return 0.1
        else:
            if self.a is None:
                x_values = [1.0-epsilon, 1.0-0.05]
                y_values = [1.0, 0.2]
                self.a, self.d = PEARL.find_a_d_symbolic(x_values, y_values)
            # adjusted reciprocal function that starts at y=1 and goes to y=0.2 at epsilon=0.05
            fraction = (self.a / (1 - epsilon + 1)) + self.d
            print(f"Epi {epi_i}, epsilon: {round(epsilon,3)}, beta: {round(beta,3)}, fraction: {round(fraction,3)}, {self.a}, {self.d}")
            return fraction

    def abstract_qlearning_episode(self, epi_i, env, agent, abstract, abstract_for_policy, evaluating=False):
        agent.clear_etable()
        state = env.reset()
        state_abs = abstract.state(state)
        rounded_state = tuple([round(x,5) for x in state])
        agent.initialize_qvalues(state_abs, init_abs_action_list=abstract.init_abs_action_list)
        abstract.add_concrete_state(state_abs, tuple(rounded_state))
        done = False
        reward = 0
        epoch = 0
        trace = Trace()
        info = {}
        info["success"] = False

        while (not done) and (epoch < env.step_max):
            ################### select an action to execute ###################
            action_abs_for_policy = abstract_for_policy.state(state)
            if evaluating:
                action_abs = agent.evaluation_policy(action_abs_for_policy, init_abs_action_list=abstract_for_policy.init_abs_action_list)
            else:
                action_abs = agent.policy(action_abs_for_policy, init_abs_action_list=abstract_for_policy.init_abs_action_list)
            
            ################### execute the action until abstract state changes ###################
            initial_state_in_abs = copy.deepcopy(state)
            initial_rounded_state_in_abs = tuple([round(x,5) for x in state])
            next_state_abs = copy.deepcopy(state_abs)
            r_abs_discounted = 0
            r_abs_total = 0
            steps_in_abs_state = 0
            discount_factor = 1.0  # discount factor for reward accumulation
            while next_state_abs == state_abs:
                ################### execute the action ###################
                action = abstract.sample_action(action_abs, env.use_mean_action)
                next_state, r, done, info = env.step(action)
                env.render()
                success = info['success']
                next_state_abs = abstract.state(next_state)
                rounded_state = tuple([round(x,5) for x in state])
                rounded_next_state = tuple([round(x,5) for x in next_state])
                # print(rounded_state, action, rounded_next_state, round(r,2), done, success)
                agent.initialize_qvalues(next_state_abs, init_abs_action_list=abstract.init_abs_action_list)
                abstract.add_concrete_state(next_state_abs, tuple(rounded_next_state))
                r_abs_total += r
                r_abs_discounted += discount_factor * r # each step's reward is discounted by gamma^(step_number)
                discount_factor *= agent._gamma  # Update discount for next step
                steps_in_abs_state += 1

                ################### compute concrete q-values and td-errors ###################
                transition = Transition(tuple(state), tuple(rounded_state), action, tuple(next_state), tuple(rounded_next_state), r, done, success, state_abs, action_abs, next_state_abs, r_abs_total, steps_in_abs_state)
                if not evaluating and abstract._bootstrap == 'from_estimated_concrete':
                    agent.estimate_concrete_qvalue(transition)
                    agent.add_measure_to_concrete_tderror(transition)
                    
                trace.append_transition(transition)
                epoch += 1
                if done or epoch >= env.step_max or tuple(rounded_state) == tuple(rounded_next_state):
                    break
                state = copy.deepcopy(next_state)
            state = copy.deepcopy(next_state)

            ################### compute abstract q-values and td-errors ###################
            if not evaluating:
                # create abstract transition
                final_rounded_state = tuple([round(x,5) for x in state])
                abstract_reward = r_abs_discounted if r_abs_discounted != 0 else r_abs_total
                abstract_transition = Transition(
                    tuple(initial_state_in_abs), initial_rounded_state_in_abs, None,
                    tuple(state), final_rounded_state,
                    r_abs_total, done, success, state_abs, action_abs, next_state_abs,
                    abstract_reward, steps_in_abs_state
                )
                agent.update_qvalue(abstract_transition)
                agent.add_measure_to_buffer(abstract_transition)

            state_abs = copy.deepcopy(next_state_abs)
            reward += r_abs_total

        ################### decay epsilon ###################
        if not evaluating:
            abstract._tree.update_n_abstract_states()
            abstraction_fraction=self.refinement_fraction(epi_i, self.agent._epsilon, self.abstract._beta, self.evaluation_mean_succ)
            self.log.log_learning(epi_i, reward, success, epoch, round(agent._epsilon,3), abstract._tree._n_abstract_states, abstract._n_abstract_actions, abstraction_fraction, self.n_unstable_states, self.n_unstable_actions)
            agent.decay_epsilon()

        return success, reward, epoch, state, state_abs, trace

    def evaluate_policy(self, epi_i, env, agent, abstract, n_epi=100):
        print(f"\nEvaluating policy at episode: {epi_i}...")
        total_succ = 0
        total_reward_list = []
        success_list = []
        steps_list = []
        succ_traces = []

        for j in range(n_epi):
            success, reward, steps, final_state, final_abs_state, trace = self.abstract_qlearning_episode(epi_i, env, agent, abstract, abstract, evaluating=True)
            total_succ += int(success)
            total_reward_list.append(reward)
            steps_list.append(steps)
            success_list.append(success)
            if success:
                if len(succ_traces) > 0 and len(trace.trace) < len(succ_traces[-1]) or len(succ_traces) == 0:
                    succ_traces = [trace.trace]

        self.log.log_evaluation(epi_i, total_reward_list, success_list, steps_list)
        mean_eval_reward, mean_eval_success, mean_eval_steps = self.log.mean_eval_reward_success_steps()
        print("Evaluation mean reward: {}".format(mean_eval_reward))
        print("Evaluation mean success: {}".format(mean_eval_success))
        print("Evaluation mean steps: {}".format(mean_eval_steps))
        if mean_eval_success[-1] > 0.9:
            trace = [str(transition) for transition in succ_traces[0]]
            print(f"Computed trace:")
            for item in trace:
                print(item)
        return np.mean(success_list)

