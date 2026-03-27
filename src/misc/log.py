import numpy as np
import matplotlib.pylab as plt
import pickle as pk
from copy import copy, deepcopy
from pathlib import Path
import os 
import csv
import tqdm
import json
np.set_printoptions(suppress=True)
from torch.utils.tensorboard import SummaryWriter

class LogExperiments:
    def __init__(self, directory):
        self._learning_data = {'reward': [], 'success': [], 'steps': [], 'episode': [], 'n_states': [], 'n_actions': [], 'epsilon': [], 'abstraction_fraction': [], 'n_unstable_states': [], 'n_unstable_actions': []}
        self._evaluation_data = {'reward_list': [], 'success_list': [], 'steps_list': [], 'episode': []}
        self._summary_writer = SummaryWriter(log_dir = directory)
        self.max_td = {}
        self.max_q = {}

    def log_learning(self, episode, reward, success, steps, epsilon, n_states, n_actions, abstraction_fraction, n_unstable_states, n_unstable_actions):
        self._learning_data['episode'].append(episode)
        self._learning_data['reward'].append(reward)
        if success: succ = 1
        else: succ = 0
        self._learning_data['success'].append(succ)
        self._learning_data['steps'].append(steps)
        self._learning_data['n_states'].append(n_states)
        self._learning_data['n_actions'].append(n_actions)
        self._learning_data['epsilon'].append(epsilon)
        self._learning_data['abstraction_fraction'].append(abstraction_fraction)
        self._learning_data['n_unstable_states'].append(n_unstable_states)
        self._learning_data['n_unstable_actions'].append(n_unstable_actions)

        self._summary_writer.add_scalar("reward", self.get_recent(self._learning_data['reward'], 100), episode)
        self._summary_writer.add_scalar("success", self.get_recent(self._learning_data['success'], 100), episode)
        self._summary_writer.add_scalar("steps", self.get_recent(self._learning_data['steps'], 100), episode)
        self._summary_writer.add_scalar("epsilon", epsilon, episode)
        self._summary_writer.add_scalar("n_states", n_states, episode)
        self._summary_writer.add_scalar("n_actions", n_actions, episode)
    
    def log_evaluation(self, episode, reward_list, success_list, steps_list):
        self._evaluation_data["episode"].append(episode)
        self._evaluation_data["reward_list"].append(reward_list)
        self._evaluation_data["success_list"].append(success_list)
        self._evaluation_data["steps_list"].append(steps_list)
        self._summary_writer.add_scalar("eval_reward_mean", np.mean(reward_list), episode)
        self._summary_writer.add_scalar("eval_success_mean", np.mean(success_list), episode)
        self._summary_writer.add_scalar("eval_steps_mean", np.mean(steps_list), episode)

    def save_execution(self, directory, file_name):
        path = directory+"/"+file_name+"_learning.pickle"
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, "wb") as output_file:
            pk.dump(self._learning_data, output_file)
        output_file.close()

        path = directory+"/"+file_name+"_evaluation.pickle"
        with open(path, "wb") as output_file:
            pk.dump(self._evaluation_data, output_file)
        output_file.close()

    def recent_mean_learning_reward(self, last):
        return self.get_recent(self._learning_data['reward'], last)
    
    def recent_mean_learning_success(self, last):
        return self.get_recent(self._learning_data['success'], last)
    
    def recent_mean_learning_steps(self, last):
        return self.get_recent(self._learning_data['steps'], last)
    
    def get_recent(self, data, last):
        size = len(data)
        if last < size: 
            x = size - last
        else: 
            x = 0
        result = data[x: size]
        return np.mean(result)
    
    def mean_eval_reward_success_steps(self):
        mean_eval_reward = [np.mean(np.array(list_)) for list_ in self._evaluation_data["reward_list"]]
        mean_eval_success = [np.mean(np.array(list_, dtype=bool)) for list_ in self._evaluation_data["success_list"]]
        mean_eval_steps = [np.mean(np.array(list_)) for list_ in self._evaluation_data["steps_list"]]
        return mean_eval_reward, mean_eval_success, mean_eval_steps

    def close_writer(self):
        self._summary_writer.close()

    def plot_result(self, moving_number, path):
        y_reward = self._learning_data["reward"]
        y_success = self._learning_data["success"]
        y_steps = self._learning_data["steps"]
        y_states = self._learning_data["n_states"]
        y_actions = self._learning_data["n_actions"]
        y_eps = self._learning_data["epsilon"][moving_number:]
        y_fraction = self._learning_data["abstraction_fraction"][moving_number:]
        n_unstable_states = self._learning_data["n_unstable_states"][moving_number:]
        n_unstable_actions = self._learning_data["n_unstable_actions"][moving_number:]
        x = self._learning_data['episode']
        x_m = []
        mean_reward = []
        mean_success = []
        mean_steps = []
        for i in range(moving_number, len(x)):
            mean_reward.append(self.get_recent(y_reward[:i], moving_number))
            mean_success.append(self.get_recent(y_success[:i], moving_number))
            mean_steps.append(self.get_recent(y_steps[:i], moving_number))
            x_m.append(i)
        fig, axs = plt.subplots(2, 4, figsize=(18, 10))

        # Plot learning curves on the first row (Reward, Success, Steps)
        axs[0, 0].plot(x_m, mean_reward, label="reward", color='r')
        axs[0, 0].set_ylabel("Learning Reward")
        axs[0, 0].set_xlabel("Episodes")
        axs[0, 0].legend()

        axs[0, 1].plot(x_m, y_eps, label="exploration", color='y')
        axs[0, 1].plot(x_m, y_fraction, label="abstraction fraction", color='m')
        axs[0, 1].plot(x_m, mean_success, label="success", color='g')
        axs[0, 1].set_ylabel("Learning Success")
        axs[0, 1].set_xlabel("Episodes")
        axs[0, 1].legend()

        axs[0, 2].plot(x_m, mean_steps, label="steps", color='b')
        axs[0, 2].set_ylabel("Learning Steps")
        axs[0, 2].set_xlabel("Episodes")
        axs[0, 2].legend()

        axs[0, 3].plot(x_m, n_unstable_states, label="#unstable_states", color="r")
        axs[0, 3].plot(x_m, n_unstable_actions, label="#unstable_actions", color="g")
        axs[0, 3].set_ylabel("# states and actions refined")
        axs[0, 3].set_xlabel("Episodes")
        axs[0, 3].legend()

        # Plot evaluation on the second row (Success Rate, Reward, Steps)
        # Evaluation: Reward
        y = self._evaluation_data["reward_list"]
        y = [np.mean(list_) for list_ in y]
        x = self._evaluation_data['episode']
        axs[1, 0].plot(x, y, label="reward", color='r')
        axs[1, 0].set_ylabel("Evaluation Reward")
        axs[1, 0].set_xlabel("Episodes")
        axs[1, 0].legend()

        # Evaluation: Success
        y = self._evaluation_data["success_list"]
        y = [np.mean(np.array(list_, dtype=bool)) for list_ in y]
        x = self._evaluation_data['episode']
        axs[1, 1].plot(x, y, label="success", color='g')
        axs[1, 1].set_ylabel("Evaluation Success")
        axs[1, 1].set_xlabel("Episodes")
        axs[1, 1].legend()

        # Evaluation: Steps
        y = self._evaluation_data["steps_list"]
        y = [np.mean(list_) for list_ in y]
        x = self._evaluation_data['episode']
        axs[1, 2].plot(x, y, label="steps", color='b')
        axs[1, 2].set_ylabel("Evaluation Steps")
        axs[1, 2].set_xlabel("Episodes")
        axs[1, 2].legend()

        # Number of States and Actions
        x = self._learning_data['episode']
        axs[1, 3].plot(x, y_states, label="#states", color='m')
        # axs[1, 3].plot(x, y_actions, label="#actions", color='y')
        axs[1, 3].set_ylabel("#States and #Actions")
        axs[1, 3].set_xlabel("Episodes")
        axs[1, 3].legend()

        plt.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def get_minm_maxm_tderror(self, eval_log):
        all_values = []
        for state in eval_log:
            for action in eval_log[state]:
                all_values.extend(eval_log[state][action])
        if len(all_values) > 0:
            minm = min(all_values)
            maxm = max(all_values)
            return minm, maxm
        return 0, 0
    
    def normalize_eval(self, eval_log, minm, maxm):
        eval = deepcopy(eval_log)
        if minm == maxm: return eval
        for state in eval_log:
            for action in eval_log[state]:
                eval[state][action] = []
                for eval_value in eval_log[state][action]: 
                    eval[state][action].append((eval_value - minm) / (maxm - minm))
        return eval
    
    def get_max_q(self, qtable):
        all_values = []
        for state in qtable:
            for action in qtable[state]:
                all_values.append(qtable[state][action])
        if len(all_values) > 0:
            minm = min(all_values)
            maxm = max(all_values)
            return minm, maxm
        return 0, 0

    def log_td(self, epi_i, td_eval, q_table, mean_disp_state=None, mean_disp_action=None):
        minm, maxm = self.get_minm_maxm_tderror(td_eval)
        self.max_td[epi_i] = maxm
        minm, maxm = self.get_max_q(q_table)
        self.max_q[epi_i] = maxm

    def print_qtable(self, q_table, dirpath, epi_i):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filepath = dirpath+f"/qtable_{epi_i}.json"

        qtable_new = {}
        for state, actions_dict in q_table._qtable.items():
            state_str = str(state)
            if state_str not in qtable_new:
                qtable_new[state_str] = {}

            sorted_action_values = sorted(actions_dict.items(), key=lambda item: item[1], reverse=True)
            qtable_new[state_str] = {str(action): value for action, value in sorted_action_values}

        with open(filepath, 'w') as file:
            json.dump(qtable_new, file, indent=1)

