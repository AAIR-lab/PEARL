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
class MulticityParamActionsEnv(gym.Env):
    def __init__(self, interactive=False):
        super().__init__()
        # layout of the map
        basepath = os.getcwd()

        # load cities

        self.city1_config = os.path.join(basepath, "environments/maps/city1.cfg")
        self.city2_config = os.path.join(basepath, "environments/maps/city2.cfg")
        self.city3_config = os.path.join(basepath, "environments/maps/city3.cfg")

    
        self.city1_obstacles, self.city1_points = map_maker.read_obstacles(self.city1_config)
        self.city2_obstacles, self.city2_points = map_maker.read_obstacles(self.city2_config)
        self.city3_obstacles, self.city3_points = map_maker.read_obstacles(self.city3_config)

        self._dimension = (1.0,1.0)
        self.interactive = interactive

        self.obstacles = { 0: self.city1_obstacles, 1: self.city2_obstacles, 2: self.city3_obstacles}
        self.points = {0: self.city1_points, 1: self.city2_points, 2: self.city3_points}
        self.current_city = 0 
        self.use_mean_action = False

        # state variables and ranges
        self.state_ranges = self.get_state_ranges()
        self.obs_low = np.array(self.state_ranges)[:,0]
        self.obs_high = np.array(self.state_ranges)[:,1]
        self.observation_space = gym.spaces.Box(low = self.obs_low, high = self.obs_high, dtype = np.float32)

        # actions and parameter ranges
        self.actions = ['up','down','left','right',"fly"]
        # self.actions = ['up','down','left','right','pickup','dropoff']
        self.action_size = len(self.actions)
        self.is_action_space_discrete = False
        self.action_param_ranges = self.get_action_param_ranges()
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(self.action_size),
                                            gym.spaces.Box(low=self.action_param_ranges[0][0][0], high=self.action_param_ranges[0][0][1], shape=(1,)),
                                            gym.spaces.Box(low=self.action_param_ranges[1][0][0], high=self.action_param_ranges[1][0][1], shape=(1,)),
                                            gym.spaces.Box(low=self.action_param_ranges[2][0][0], high=self.action_param_ranges[2][0][1], shape=(1,)),
                                            gym.spaces.Box(low=self.action_param_ranges[3][0][0], high=self.action_param_ranges[3][0][1], shape=(1,)),
                                            gym.spaces.Discrete(3)
                                    ))
        self._action_probs = {0:[2,3], 1:[2,3], 2:[0,1], 3:[0,1], 4:[4], 5:[5]}

        self.proximity_dist = 0.06
        self._stoch_prob = 0.9
        self.fixed_values = [[0, 0], [1, 0], [0, 1], [1, 1]] # used for plotting abstractions
        self.initial_refinement_order = [[1,1,1,1], [0,0,1,1]]
        # [[1,1] + [0 for i in range(2, len(self.state_ranges)-1)] + [1], [0,0] +[1 for i in range(2, len(self.state_ranges)-1)] + [0]]

    def initialize_problem(self, start_pos, agent_city, package_pos, package_city, target_pos, target_city, airport_city1, airport_city2, airport_city3, step_max):
        '''
            sets initial taxi location, passenger pickup location, and intaxi=0 in the initial state
        '''
        # start, agent_city, package, package_city,  goal, target_city = args["start_pos"], args["agent_city"], args["package_pos"], args["package_city"],args["target_pos"], args["target_city"]
        start, agent_city, package, package_city,  goal, target_city = start_pos, agent_city, package_pos, package_city, target_pos, target_city
        # airport_city1, airpot_city2, airport_city3 = args["airport_city1"], args["airport_city2"], args["airport_city3"]
        airport_city1, airport_city2, airport_city3 = airport_city1, airport_city2, airport_city3

        self.airport_locations = {0: airport_city1, 1: airport_city2, 2: airport_city3}

        self._init_agent_loc = copy.deepcopy(start)
        self._init_agent_city = copy.deepcopy(agent_city)

        self.packge_loc = copy.deepcopy(package)
        self.package_city = copy.deepcopy(package_city)

        self._target_loc = copy.deepcopy(goal)
        self._target_city = copy.deepcopy(target_city)

        self.current_city = self._init_agent_city
        self._has_package = 0 
        self._at_target = 0 

        self._init_state = list(self._init_agent_loc) + [self._has_package] + [self.current_city]
        self._goal_state = list(goal) + [1] + [0]
        self.reset()
        self.step_max = step_max

        # Launch interactive pygame
        if self.interactive:
            pygame.init()
            pygame.display.set_caption('Office Domain')
            width, height = 1500, 500
            self.screen = pygame.display.set_mode((width, height))
            self.environment_view = RenderView(self.screen, self.points, self._init_state, self.current_city, self.packge_loc, self.package_city, self._target_loc, self._target_city, self.airport_locations)
        print()

    def get_state_ranges(self):
        ### The origin is the top left corner of the map

        # state ranges
        ranges = [ [np.float32(0), np.float32(self._dimension[0])],   # robot y
                   [np.float32(0), np.float32(self._dimension[1])],   # robot x
                   [np.float32(0), np.float32(1)],                    # has item
                   [np.float32(0), np.float32(3)],                    # which city
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
        action_param_ranges[4] = [[0,3]]
        
        self.is_int_discrete_action_params = {}
        self.is_int_discrete_action_params[0] = False
        self.is_int_discrete_action_params[1] = False
        self.is_int_discrete_action_params[2] = False
        self.is_int_discrete_action_params[3] = False 
        self.is_int_discrete_action_params[4] = True
        # self.is_int_discrete_action_params[4] = None
        # self.is_int_discrete_action_params[5] = None    
        return action_param_ranges


    def reset(self):
        self.steps = 0
        self._agent_loc = copy.deepcopy(self._init_agent_loc)
        self._has_package = 0
        self._at_target = 0 
        self.current_city = copy.deepcopy(self._init_agent_city)
        self._state = copy.deepcopy(self._init_state)
        return self._state


    def step(self, action_input):
        move_cost = -1
        collision_cost = -2
        # wrong_pickup_dropoff_cost = -100
        goal_reward = 10000 #30000
        self.steps += 1
        reward = 0 
        done = False 
        success = False
        flew = False

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
        elif discrete_action == 4 and self._near_airpot(): # fly 
            self.current_city = int(action_params[0]) 
            next_loc = random.choice(self.airport_locations[self.current_city])
            flew = True
        else:
            next_loc = copy.deepcopy(self._agent_loc)

        if flew:
            self._agent_loc = copy.deepcopy(next_loc)
            reward += move_cost
        else:
            if not self.within_bounds(next_loc) or self.collision_point(next_loc[0], next_loc[1]) or self.colliding(self._agent_loc, next_loc):
                reward += collision_cost
            else:
                self._agent_loc = copy.deepcopy(next_loc)
                if self._has_package == 0 and self.at_package_loc(): 
                    self._has_package = 1 
                elif self._has_package == 1 and self.at_drop_loc():
                    self._at_drop = 1 
                    done = True 
                    success = True 
                    reward += goal_reward 
                else: 
                    reward += move_cost

        self._state = list(self._agent_loc) + [self._has_package] + [self.current_city]
        if self.steps >= self.step_max:
            done = True

        # print(f"action: {action} state {self._state}")
        return self._state, reward, done, {"success": success, "steps": self.steps}


    def _near_airpot(self):
        for ap in self.airport_locations[self.current_city]:
            if abs(ap[0] - self._agent_loc[0]) <= self.proximity_dist and abs(ap[1] - self._agent_loc[1]) <= self.proximity_dist:
                return True
        return False
    def action_stochastic(self, action_index):
        if random.uniform(0,1) > self._stoch_prob:
            action_index_stoch = random.choice(self._action_probs[action_index])
        else: 
            action_index_stoch = action_index
        return action_index_stoch

########################## additional utility functions ##########################
    def at_package_loc(self):
        if self.current_city == self.package_city:
            return abs(self.packge_loc[0] - self._agent_loc[0]) <= self.proximity_dist and abs(self.packge_loc[1] - self._agent_loc[1]) <= self.proximity_dist 
        return False
 
    def at_drop_loc(self):
        if self.current_city == self._target_city:
            return abs(self._target_loc[0] - self._agent_loc[0]) <= self.proximity_dist and abs(self._target_loc[1] - self._agent_loc[1]) <= self.proximity_dist
        return False

    def within_bounds(self, loc):
        # check if a location is within the env bound        
        flag = False
        if loc[0] < self._dimension[0] and loc[0] >= 0:
            if loc[1] < self._dimension[1] and loc[1] >= 0:
                flag = True
        return flag

    def collision_point(self, x, y):
        ballpoint = shapely.geometry.Point((x,y))
        ballcircle = ballpoint.buffer(0.03)
        for obs in self.obstacles[self.current_city]:
            # if ballcircle.intersects(self.obstacle):
            if obs.contains(ballpoint) or ballcircle.intersects(obs) or obs.intersects(ballcircle):
                return True
        return False

    def colliding(self, point, new_point):
        line = shapely.geometry.LineString([point, new_point])
        for obs in self.obstacles[self.current_city]:
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
            self.environment_view.blit(self._agent_loc, self._has_package, self.current_city)
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
    def __init__(self, screen, obs_points, ball_pos, agent_city, package_pos, package_city, target_pos, target_city, airport_locations):
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
        self.BALL_COLOR_package = [0, 255, 255]
        self.AIRPORT_COLOR = [255,255,0]
        self.TARGET_COLOR = [255, 0, 0]
        self.package_pos = package_pos
        self.package_city = package_city
        self.target_pos = target_pos
        self.target_city = target_city
        self.agent_pos = ball_pos
        self.agent_city = agent_city

        self.height = self.screen.get_height()
        self.width = self.screen.get_width() / 3.0

        self.airport_locations = airport_locations

        self.obs_points = obs_points

        # Draw the background
        self.screen.fill((255, 255, 255))

        for city in self.obs_points: 
            polys = self.obs_points[city]
            for poly in polys:
                points = [self._to_pixels(o,city) for o in poly]
                pygame.draw.polygon(self.screen, self.DARK_GRAY, points, 0)

        for city in self.airport_locations: 
            for ap in self.airport_locations[city]: 
                point = self._to_pixels(ap,city)
                pygame.draw.circle(self.screen,self.AIRPORT_COLOR, point, int(0.02*self.width) )

        pygame.draw.circle(self.screen, self.BALL_COLOR, self._to_pixels(ball_pos,self.agent_city), int(0.02*self.width))
        pygame.draw.circle(self.screen, self.BALL_COLOR_package, self._to_pixels(self.package_pos,self.package_city), int(0.02*self.width))
        pygame.draw.circle(self.screen, self.TARGET_COLOR, self._to_pixels(self.target_pos,self.target_city), int(0.02*self.width))

        pygame.display.flip()


    def _to_pixels(self, pt,city):
        """ Converts from real units in the 0-1 range to pixel units

        :param pt: a point in real units
        :type pt: list
        :returns: the input point in pixel units
        :rtype: list

        """
        offset = self.width * city
        return [int(pt[0] * self.width) + offset, int(pt[1] * self.height)]

    def blit(self, agent_pos, has_package, current_city):
        """ Blit the ball onto the background surface """
        self.screen.fill((255, 255, 255))
        for city in self.obs_points: 
            polys = self.obs_points[city]
            for poly in polys:
                points = [self._to_pixels(o,city) for o in poly]
                pygame.draw.polygon(self.screen, self.DARK_GRAY, points, 0)
        
        for city in self.airport_locations: 
            for ap in self.airport_locations[city]: 
                point = self._to_pixels(ap,city)
                pygame.draw.circle(self.screen,self.AIRPORT_COLOR, point, int(0.02*self.width) )

        pygame.draw.circle(self.screen, self.TARGET_COLOR, self._to_pixels(self.target_pos,self.target_city), int(0.02*self.width))
        pygame.draw.circle(self.screen, self.BALL_COLOR_package, self._to_pixels(self.package_pos,self.package_city), int(0.02*self.width))

        if has_package: 
            ball_color = self.BALL_COLOR_package
        else:
            ball_color = self.BALL_COLOR
        pygame.draw.circle(self.screen, ball_color, self._to_pixels(agent_pos,current_city), int(0.02*self.width))
        pygame.display.flip()
