import os
import sys
import numpy as np
import random 
import copy
import gymnasium as gym
import shapely
import environments.maps.map_maker as map_maker

try:
    import pygame
except ImportError:
    print('Pygame not available ')

"""
Continuous state variables: x, y, has_coffee, has_mail
Parameterized actions: up, down, left, right
"""

class OfficeParamActionsEnv(gym.Env):
    def __init__(self, map_name, interactive=False):
        super().__init__()
        # layout of the map
        basepath = os.getcwd()
        self.configuration = f"{basepath}/environments/maps/"+map_name
        self.obstacles, self.points = map_maker.read_obstacles(self.configuration)
        self._dimension = (1.0,1.0)
        self.interactive = interactive

        # state variables and ranges
        self.state_ranges = self.get_state_ranges()
        self.obs_low = np.array(self.state_ranges)[:,0]
        self.obs_high = np.array(self.state_ranges)[:,1]
        self.observation_space = gym.spaces.Box(low = self.obs_low, high = self.obs_high, dtype = np.float32)

        # actions and parameter ranges
        self.actions = ['up','down','left','right']
        # self.actions = ['up','down','left','right','pickup','dropoff']
        self.action_size = len(self.actions)
        self.is_action_space_discrete = False
        self.action_param_ranges = self.get_action_param_ranges()
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(self.action_size),
                                            gym.spaces.Box(low=self.action_param_ranges[0][0][0], high=self.action_param_ranges[0][0][1], shape=(1,)),
                                            gym.spaces.Box(low=self.action_param_ranges[1][0][0], high=self.action_param_ranges[1][0][1], shape=(1,)),
                                            gym.spaces.Box(low=self.action_param_ranges[2][0][0], high=self.action_param_ranges[2][0][1], shape=(1,)),
                                            gym.spaces.Box(low=self.action_param_ranges[3][0][0], high=self.action_param_ranges[3][0][1], shape=(1,)),
                                    ))
        self._action_probs = {0:[2,3], 1:[2,3], 2:[0,1], 3:[0,1], 4:[4], 5:[5]}

        self.proximity_dist = 0.06
        self._stoch_prob = 0.9
        self.fixed_values = [[0, 0], [1, 0], [0, 1], [1, 1]] # used for plotting abstractions
        self.initial_refinement_order = [[1,1] + [0 for i in range(2, len(self.state_ranges))], [0,0] +[1 for i in range(2, len(self.state_ranges))]]
        self.use_mean_action = False

    def initialize_problem(self, start_pos, coffee_pos, mail_pos, target_pos, step_max):
        '''
            sets initial taxi location, passenger pickup location, and intaxi=0 in the initial state
        '''
        # start, coffee, mail, goal = args["start_pos"], args["coffee_pos"], args["mail_pos"], args["target_pos"]
        start, coffee, mail, goal = start_pos, coffee_pos, mail_pos, target_pos
        self._init_agent_loc = copy.deepcopy(start)
        self._coffee_loc = copy.deepcopy(coffee)
        self._mail_loc = copy.deepcopy(mail)
        self._target_loc = copy.deepcopy(goal)
        self._at_coffee = 0
        self._at_mail = 0
        self._has_coffee = 0
        self._has_mail = 0
        self._has_coffee_mail = 0
        self._init_state = list(self._init_agent_loc) + [self._has_coffee] + [self._has_mail]
        self._goal_state = list(goal) + [1] + [1]
        self.reset()
        self.step_max = step_max
        self.object_locs = [self._coffee_loc, self._mail_loc]

        # Launch interactive pygame
        if self.interactive:
            pygame.init()
            pygame.display.set_caption('Office Domain')
            width, height = 500, 500
            self.screen = pygame.display.set_mode((width, height))
            self.environment_view = RenderView(self.screen, self.points, self._init_state, self._coffee_loc, self._mail_loc, self._goal_state)
        print()

    def get_state_ranges(self):
        ### The origin is the top left corner of the map

        # state ranges
        ranges = [ [np.float32(0), np.float32(self._dimension[0])],   # robot y
                   [np.float32(0), np.float32(self._dimension[1])],   # robot x
                   [np.float32(0), np.float32(1)],                    # has coffee
                   [np.float32(0), np.float32(1)],                    # has mail
                 ]
        self._original_vars = [i for i in range(0,len(ranges))]

        self.is_int_state_variable = [False, False, True, True]
        return ranges


    def get_action_param_ranges(self):
        # action param ranges
        action_param_ranges = {}
        action_param_ranges[0] = [[-0.05,0.0]]   # up
        action_param_ranges[1] = [[0.0,0.05]]    # down
        action_param_ranges[2] = [[-0.05,0.0]]   # left
        action_param_ranges[3] = [[0.0,0.05]]    # right
        # action_param_ranges[4] = []              # pickup
        # action_param_ranges[5] = []              # dropoff
        
        self.is_int_discrete_action_params = {}
        self.is_int_discrete_action_params[0] = False
        self.is_int_discrete_action_params[1] = False
        self.is_int_discrete_action_params[2] = False
        self.is_int_discrete_action_params[3] = False 
        # self.is_int_discrete_action_params[4] = None
        # self.is_int_discrete_action_params[5] = None    
        return action_param_ranges


    def reset(self):
        self.steps = 0
        self._agent_loc = copy.deepcopy(self._init_agent_loc)
        self._at_coffee = 0
        self._at_mail = 0
        self._has_coffee = 0
        self._has_mail = 0
        self._has_coffee_mail = 0
        self._state = copy.deepcopy(self._init_state)
        return self._state


    def step(self, action_input):
        move_cost = -1
        collision_cost = -2
        # wrong_pickup_dropoff_cost = -100
        goal_reward = 1000
        self.steps += 1
        reward = 0 
        done = False 
        success = False

        action = self.action_stochastic(action_input[0])
        action = [action, action_input[1]]
        discrete_action = action[0]
        action_params = action[1]

        if discrete_action in [0,1]: # move up or down
            dy = action_params[0]
            next_loc = [self._agent_loc[0]+dy, self._agent_loc[1]]
        elif discrete_action in [2,3]: # move left or right
            dx = action_params[0]
            next_loc = [self._agent_loc[0], self._agent_loc[1]+dx]
            
        if not self.within_bounds(next_loc) or self.collision_point(next_loc[0], next_loc[1]) or self.colliding(self._agent_loc, next_loc):
            reward += collision_cost
        else:
            self._agent_loc = copy.deepcopy(next_loc)
            if self._has_coffee == 0 and self.at_coffee_loc():
                self._has_coffee = 1
            elif self._has_mail == 0 and self.at_mail_loc():
                self._has_mail = 1
            elif self.coffee_and_mail_at_target():
                done = True
                success = True
                reward += goal_reward
            else:
                reward += move_cost

        # elif action[0] == 4: # pickup
        #     if self._has_coffee == 0 and self.at_coffee_loc():
        #         self._has_coffee = 1
        #     elif self._has_mail == 0 and self.at_mail_loc():
        #         self._has_mail = 1
        #     else:
        #         reward += wrong_pickup_dropoff_cost

        # elif action[0] == 5: # dropoff
        #     if self.coffee_and_mail_at_target():
        #         self._has_coffee = 0
        #         self._has_mail = 0
        #         done = True
        #         success = True
        #         reward += goal_reward
        #     else:
        #         reward += wrong_pickup_dropoff_cost

        if self.at_coffee_loc():
            self._at_coffee = 1
        else:
            self._at_coffee = 0

        if self.at_mail_loc():
            self._at_mail = 1
        else:
            self._at_mail = 0

        self._state = list(self._agent_loc) + [self._has_coffee] + [self._has_mail]
        if self.steps >= self.step_max:
            done = True 

        # print(f"action: {action} state {self._state}")
        return self._state, reward, done, {"success": success, "steps": self.steps}

    def action_stochastic(self, action_index):
        if random.uniform(0,1) > self._stoch_prob:
            action_index_stoch = random.choice(self._action_probs[action_index])
        else: 
            action_index_stoch = action_index
        return action_index_stoch

########################## additional utility functions ##########################
    def at_coffee_loc(self):
        return abs(self._coffee_loc[0] - self._agent_loc[0]) <= self.proximity_dist and abs(self._coffee_loc[1] - self._agent_loc[1]) <= self.proximity_dist 

    def at_mail_loc(self):
        return abs(self._mail_loc[0] - self._agent_loc[0]) <= self.proximity_dist and abs(self._mail_loc[1] - self._agent_loc[1]) <= self.proximity_dist 

    def at_target(self):
        return abs(self._target_loc[0] - self._agent_loc[0]) <= self.proximity_dist and abs(self._target_loc[1] - self._agent_loc[1]) <= self.proximity_dist

    def coffee_and_mail_at_target(self):
        return self.at_target() and self._has_coffee == 1 and self._has_mail == 1

    def within_bounds(self, loc):
        # check if a location is within the env bound        
        flag = False
        if loc[0] < self._dimension[0] and loc[0] >= 0:
            if loc[1] < self._dimension[1] and loc[1] >= 0:
                flag = True
        return flag

    def collision_point(self, x, y):
        ballpoint = shapely.geometry.Point((x,y))
        # ballcircle = ballpoint.buffer(0.01)
        for obs in self.obstacles:
            # if ballcircle.intersects(self.obstacle):
            if obs.contains(ballpoint):
                return True
        return False

    def colliding(self, point, new_point):
        line = shapely.geometry.LineString([point, new_point])
        for obs in self.obstacles:
            if shapely.intersects(line, obs):
                return True
        return False

    def state_to_index (self, state):
        return tuple(state)

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

################################## render functions ##################################

    def render(self, mode='human'):
        if self.interactive:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.environment_view.blit(self._agent_loc, self._has_coffee, self._has_mail)
            pygame.display.flip()

    def close(self):
        if self.interactive:
            pygame.quit()

class RenderView:
    """ This class displays a :class:`PinballModel`

    This class is used in conjunction with the :func:`run_pinballview`
    function, acting as a *controller*.

    We use `pygame <http://www.pygame.org/>` to draw the environment.

    """
    def __init__(self, screen, obs_points, ball_pos, coffee_pos, mail_pos, target_pos):
        """
        :param screen: a pygame surface
        :type screen: :class:`pygame.Surface`
        :param model: an instance of a :class:`PinballModel`
        :type model: :class:`PinballModel`
        """
        self.screen = screen

        self.DARK_GRAY = [64, 64, 64]
        self.LIGHT_GRAY = [232, 232, 232]
        self.BALL_COLOR = [0, 0, 255]
        self.BALL_COLOR_coffee = [0, 255, 255]
        self.BALL_COLOR_mail = [255, 255, 0]
        self.BALL_COLOR_coffee_mail = [0, 255, 0]
        self.COFFEE_COLOR = [0, 255, 255]
        self.MAIL_COLOR = [255, 255, 0]
        self.TARGET_COLOR = [255, 0, 0]
        self.coffee_pos = coffee_pos
        self.mail_pos = mail_pos

        self.obs_points = obs_points
        self.target_pos = target_pos

        # Draw the background
        self.screen.fill((255, 255, 255))

        for points in self.obs_points:
            points = [self._to_pixels(o) for o in points]
            pygame.draw.polygon(self.screen, self.DARK_GRAY, points, 0)

        pygame.draw.circle(self.screen, self.BALL_COLOR, self._to_pixels(ball_pos), int(0.02*self.screen.get_width()))
        pygame.draw.circle(self.screen, self.COFFEE_COLOR, self._to_pixels(self.coffee_pos), int(0.02*self.screen.get_width()))
        pygame.draw.circle(self.screen, self.MAIL_COLOR, self._to_pixels(self.mail_pos), int(0.02*self.screen.get_width()))
        pygame.draw.circle(self.screen, self.TARGET_COLOR, self._to_pixels(self.target_pos), int(0.02*self.screen.get_width()))

        pygame.display.flip()


    def _to_pixels(self, pt):
        """ Converts from real units in the 0-1 range to pixel units

        :param pt: a point in real units
        :type pt: list
        :returns: the input point in pixel units
        :rtype: list

        """
        return [int(pt[0] * self.screen.get_width()), int(pt[1] * self.screen.get_height())]

    def blit(self, agent_pos, has_coffee, has_mail):
        """ Blit the ball onto the background surface """
        self.screen.fill((255, 255, 255))
        for points in self.obs_points:
            points = [self._to_pixels(o) for o in points]
            pygame.draw.polygon(self.screen, self.DARK_GRAY, points, 0)
        pygame.draw.circle(self.screen, self.TARGET_COLOR, self._to_pixels(self.target_pos), int(0.02*self.screen.get_width()))
        pygame.draw.circle(self.screen, self.COFFEE_COLOR, self._to_pixels(self.coffee_pos), int(0.02*self.screen.get_width()))
        pygame.draw.circle(self.screen, self.MAIL_COLOR, self._to_pixels(self.mail_pos), int(0.02*self.screen.get_width()))
        if has_coffee and not has_mail:
            ball_color = self.BALL_COLOR_coffee
        elif has_mail and not has_coffee:
            ball_color = self.BALL_COLOR_mail
        elif has_coffee and has_mail:
            ball_color = self.BALL_COLOR_coffee_mail
        else:
            ball_color = self.BALL_COLOR
        pygame.draw.circle(self.screen, ball_color, self._to_pixels(agent_pos), int(0.02*self.screen.get_width()))
        pygame.display.flip()
