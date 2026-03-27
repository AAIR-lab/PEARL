import numpy as np
import cv2
import math
import copy
import os
import matplotlib.pyplot as plt


class VisualizeAbstraction():

    def __init__(self, env, agent, tree, directory, abstraction_directory, plot_abstractions, state_method, create_directory_method):
        self._env = env
        self._agent = agent
        self._tree = tree
        self._directory = directory
        self.abstraction_directory = abstraction_directory
        self._plot_abstractions = plot_abstractions
        self.state = state_method
        self.create_abstraction_directory = create_directory_method

    def initialize_decision_boundaries(self):
        self.id_ = 0
        if self._plot_abstractions:
            _, self.ax = plt.subplots(figsize=(4, 3))
        self.i = 0
        self.last_used_id = 0
        self.scale = 1
        if self._plot_abstractions:
            if self._env._dimension[0] == 1.0: #keep this small for large map
                self.n = 400 #200
            else:
                self.n = 20
            width, height = int(self._env._dimension[0]*self.scale*self.n), int(self._env._dimension[1]*self.scale*self.n)
            self.opacity_range = (50,250)
            self.default_color = [255, 255, 255, self.opacity_range[1]] #BGRA A=50 is darker than A=250 (this range of 50-250 for alpha seems better for visualization)

            self.img = {}
            self.img_id = {}
            for n in range(len(self._env.fixed_values)):
                fixed_values = self._env.fixed_values[n]
                self.img[n] = np.full(shape = (width,height,4), fill_value = self.default_color, dtype = np.float32)
                self.img_id[n] = np.zeros(shape = (width,height))
                for i in np.arange(0.001,self._env._dimension[0]-0.001,self._env._dimension[0]/self.n): 
                    for j in np.arange(0.001,self._env._dimension[1]-0.001,self._env._dimension[1]/self.n): 
                        if self._env.collision_point(i,j):
                            # print((i,j)) 
                            start_i, end_i = int(i*self.scale*self.n), int((i+(self._env._dimension[0]/self.n))*self.scale*self.n)
                            start_j, end_j = int(j*self.scale*self.n), int((j+(self._env._dimension[1]/self.n))*self.scale*self.n)

                            self.img_id[n][start_i:end_i, start_j:end_j] = -10 
                            self.img[n][start_i:end_i, start_j:end_j] = np.array([0,0,0,self.default_color[3]])
                            # print(start_i, end_i, start_j, end_j)

                self.create_abstraction_directory()
                # print(fixed_values)
                cv2.imwrite(f"{self._directory}/{self.abstraction_directory}/img_epi_0_abs_{len(self._tree._leaves)}_{fixed_values}.png", self.img[n])

    def update_decision_boundaries(self, clf, parent_id, last_used_id, uniform_partitioning=False):
        for n in range(len(self._env.fixed_values)):
            fixed_values = self._env.fixed_values[n]
            pred_to_id = {}
            for i,j in np.argwhere(self.img_id[n] == parent_id):
                if self.img_id[n][i,j] != -10:
                    if uniform_partitioning:
                        norm_x, norm_y = (i*self._env._dimension[0]) / (self.img[n].shape[0]*self.scale), (j*self._env._dimension[1]) / (self.img[n].shape[1]*self.scale)
                        abs_state = self.state([norm_x, norm_y]+fixed_values)
                        if abs_state is not None:
                            self.img_id[n][i,j] = abs_state.id
                    else:
                        norm_x, norm_y = (i) / (self.img[n].shape[0]*self.scale), (j) / (self.img[n].shape[1]*self.scale)
                        if clf is None:
                            pred = 0
                        else:
                            pred = clf.predict([[norm_x, norm_y]+fixed_values])
                            pred = pred[0]
                        if pred not in pred_to_id:
                            pred_to_id[pred] = last_used_id+pred+1
                        self.img_id[n][i,j] = pred_to_id[pred]

    def plot_decision_boundaries(self, filename=None):
        # abs_state ids should be set in self.img_id
        # assigns color in self.img
        print("Min qvalue: ", self._agent.min_qvalue)
        print("Max qvalue: ", self._agent.max_qvalue)
        for n in range(len(self._env.fixed_values)):
            fixed_values = self._env.fixed_values[n]
            pred_to_color = {}
            # R, G, B, Y, Purple, Teal
            # BGR color
            action_to_color = {0: [0,0,255], 1: [0,255,0], 2: [255,0,0], 3: [0,200,255], 4: [200,0,255], 5: [255,200,0]}

            for i in np.arange(0.001,self._env._dimension[0]-0.001,self._env._dimension[0]/self.n): 
                for j in np.arange(0.001,self._env._dimension[1]-0.001,self._env._dimension[1]/self.n): 
                    if not self._env.collision_point(i,j):
                        start_i, end_i = int(i*self.scale*self.n), int((i+(self._env._dimension[0]/self.n))*self.scale*self.n)
                        start_j, end_j = int(j*self.scale*self.n), int((j+(self._env._dimension[1]/self.n))*self.scale*self.n)
                        self.img_id[n][start_i:end_i, start_j:end_j] = self.img_id[n][start_i,start_j]

            for abs_state in self._tree._leaves:
                state_id = abs_state.id
                for i,j in np.argwhere(self.img_id[n] == state_id):
                    if self.img_id[n][i,j] != -10:
                        if abs_state not in pred_to_color:
                            if self._agent.min_qvalue == math.inf or self._agent.max_qvalue == -math.inf or abs_state not in self._agent._qtable._qtable:
                                color = copy.deepcopy(self.default_color)
                            else:
                                best_action = self._agent._qtable.get_best_action(abs_state, self._agent.rng_eval)
                                if best_action is not None:
                                    best_qvalue = self._agent._qtable.get_max_qvalue(abs_state, 0.0)
                                    opacity = ((best_qvalue - self._agent.min_qvalue) / (self._agent.max_qvalue - self._agent.min_qvalue))
                                    opacity = self.opacity_range[0] + (1.0 - opacity) * (self.opacity_range[1] - self.opacity_range[0])
                                    color = np.array(action_to_color[best_action.discrete_action] + [opacity])     
                                else:
                                    color = self.default_color                    
                            pred_to_color[abs_state] = color
                        self.img[n][i,j,:] = pred_to_color[abs_state]

            # Detect boundaries and apply the boundary color
            boundary_color = np.array([51, 0, 102, 1])
            for i in range(1, self.img[n].shape[0] - 1):
                for j in range(1, self.img[n].shape[1] - 1):
                    if self.img_id[n][i, j] != self.img_id[n][i - 1, j] and self.img_id[n][i, j] != -10 and self.img_id[n][i - 1, j] != -10:
                        self.img[n][i, j, :] = boundary_color
                    elif self.img_id[n][i, j] != self.img_id[n][i + 1, j] and self.img_id[n][i, j] != -10 and self.img_id[n][i + 1, j] != -10:
                        self.img[n][i, j, :] = boundary_color
                    elif self.img_id[n][i, j] != self.img_id[n][i, j - 1] and self.img_id[n][i, j] != -10 and self.img_id[n][i, j - 1] != -10:
                        self.img[n][i, j, :] = boundary_color
                    elif self.img_id[n][i, j] != self.img_id[n][i, j + 1] and self.img_id[n][i, j] != -10 and self.img_id[n][i, j + 1] != -10:
                        self.img[n][i, j, :] = boundary_color

            self.start_goal_object_marker(self._env._init_state, self._env._goal_state, self._env.object_locs, n)
            if filename is None:
                filename = f"img_abs_{len(self._tree._leaves)}.png"
            self.create_abstraction_directory()
            cv2.imwrite(f"{self._directory}/{self.abstraction_directory}/{filename}_{fixed_values}.png", self.img[n])

    def start_goal_object_marker(self, start_pos, goal_pos, object_locs, n):
        h = len(self.img[n])
        w = len(self.img[n][0])
        s = start_pos
        g = goal_pos

        i,j = s[0], s[1]
        start_i, end_i = int(i*self.scale*self.n), int((i+self._env._dimension[0]/self.n)*self.scale*self.n)
        start_j, end_j = int(j*self.scale*self.n), int((j+self._env._dimension[1]/self.n)*self.scale*self.n)

        self.img[n][start_i:end_i, start_j:end_j] = np.array([255,255,255,self.default_color[3]])

        i,j = g[0], g[1]
        start_i, end_i = int(i*self.scale*self.n), int((i+self._env._dimension[0]/self.n)*self.scale*self.n)
        start_j, end_j = int(j*self.scale*self.n), int((j+self._env._dimension[1]/self.n)*self.scale*self.n)

        self.img[n][start_i:end_i, start_j:end_j] = np.array([255,0,255,self.default_color[3]])   

        for loc in object_locs:
            i,j = loc[0], loc[1]
            start_i, end_i = int(i*self.scale*self.n), int((i+self._env._dimension[0]/self.n)*self.scale*self.n)
            start_j, end_j = int(j*self.scale*self.n), int((j+self._env._dimension[1]/self.n)*self.scale*self.n)

            self.img[n][start_i:end_i, start_j:end_j] = np.array([204,204,0,self.default_color[3]])   
          
 