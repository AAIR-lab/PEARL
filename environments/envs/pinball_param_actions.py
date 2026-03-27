from environments.envs.pinball.pinball import *
import copy
import sys
import gymnasium as gym
import numpy as np
from src.misc import utils
import shapely

"""
Continuous state variables: self.ball.position[0], self.ball.position[1], self.ball.xdot, self.ball.ydot
Parameterized actions: 'acc_x', 'acc_y', 'dec_x', 'dec_y', 'acc_none'
"""

class PinballEnv(gym.Env):
    def __init__(self, map_name, interactive=False):
        basepath = os.getcwd()
        self.configuration = f"{basepath}/environments/maps/"+map_name
        self._dimension = (1,1)
        self.interactive = interactive

        # state variables and ranges
        self.state_ranges = self.get_state_ranges()
        self.obs_low = np.array(self.state_ranges)[:,0]
        self.obs_high = np.array(self.state_ranges)[:,1]
        self.observation_space = gym.spaces.Box(low = self.obs_low, high = self.obs_high, dtype = np.float32)

        # actions and parameter ranges
        self.actions = ['acc_x', 'acc_y', 'dec_x', 'dec_y', 'acc_none']
        self.action_size = len(self.actions)
        self.is_action_space_discrete = False
        self.action_param_ranges = self.get_action_param_ranges()
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(self.action_size),
                                            gym.spaces.Box(low=self.action_param_ranges[0][0][0], high=self.action_param_ranges[0][0][1], shape=(1,)),
                                            gym.spaces.Box(low=self.action_param_ranges[1][0][0], high=self.action_param_ranges[1][0][1], shape=(1,)),
                                            gym.spaces.Box(low=self.action_param_ranges[2][0][0], high=self.action_param_ranges[2][0][1], shape=(1,)),
                                            gym.spaces.Box(low=self.action_param_ranges[3][0][0], high=self.action_param_ranges[3][0][1], shape=(1,)),
                                            gym.spaces.Box(low=self.action_param_ranges[4][0][0], high=self.action_param_ranges[4][0][1], shape=(1,)),
                                        ))

        self.fixed_values = [[0, 0]] # used for plotting abstractions
        self.initial_refinement_order = [[1,1,1,1]]
        self.use_mean_action = True
        
    def initialize_problem(self, step_max):
        self.pinball_initial = PinballModel(self.configuration)
        self.pinball = copy.deepcopy(self.pinball_initial)
        self.reset()
        self.step_max = step_max
        self._init_state = copy.deepcopy(self.pinball.ball.position)
        self._goal_state = copy.deepcopy(self.pinball.target_pos)
        self.object_locs = []
        print("Initial state: ", self._init_state)
        print("Goal state: ", self._goal_state)

        # Launch interactive pygame
        if self.interactive:
            try:
                import pygame
            except ImportError:
                print('Pygame not available ')

            pygame.init()
            width, height = 500, 500 #self._maze.shape[0]*100, self._maze.shape[1]*100
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption('Pygame Visualization')
            self.environment_view = PinballView(self.screen, self.pinball)
        print()

    def reset(self):
        self.steps = 0
        self.pinball = copy.deepcopy(self.pinball_initial)
        state = self.pinball.get_state()
        return state

    def get_state_ranges(self):
        # [self.ball.position[0], self.ball.position[1], self.ball.xdot, self.ball.ydot]
        state_ranges = []
        state_ranges.append([np.float32(0.0), np.float32(1.0)])
        state_ranges.append([np.float32(0.0), np.float32(1.0)])
        state_ranges.append([np.float32(-2.0), np.float32(2.0)])
        state_ranges.append([np.float32(-2.0), np.float32(2.0)])

        self.is_int_state_variable = [False for i in range(len(state_ranges))]
        return state_ranges

    def get_action_param_ranges(self):
        action_param_ranges = {}
        action_param_ranges[0] = [[0.0, 1.0]]
        action_param_ranges[1] = [[0.0, 1.0]]
        action_param_ranges[2] = [[-1.0, 0.0]]
        action_param_ranges[3] = [[-1.0, 0.0]]
        action_param_ranges[4] = [[0.0, 0.0]]

        self.is_int_discrete_action_params = {}
        self.is_int_discrete_action_params[0] = False 
        self.is_int_discrete_action_params[1] = False 
        self.is_int_discrete_action_params[2] = False 
        self.is_int_discrete_action_params[3] = False 
        self.is_int_discrete_action_params[4] = True 
        return action_param_ranges
    
    def step(self, action):
        discrete_action = action[0]
        action_params = action[1]
        if discrete_action in [0,2]:
            self.pinball.action_effects[discrete_action][0] = action_params[0]
        elif discrete_action in [1,3]:
            self.pinball.action_effects[discrete_action][1] = action_params[0]
        reward, done, success = self.pinball.take_action(discrete_action)
        self._check_bounds_velocity()
        state = self.pinball.get_state()
        self.steps += 1
        info = {}
        info["success"] = success
        info["steps"] = self.steps
        if self.steps >= self.step_max:
            done = True 
        return state, reward, done, info

    def _check_bounds_velocity(self):
        """ Make sure that the ball stays within the environment """
        if self.pinball.ball.xdot > self.state_ranges[2][1]:
            self.pinball.ball.xdot = self.state_ranges[2][1]
        if self.pinball.ball.xdot < self.state_ranges[2][0]:
            self.pinball.ball.xdot = self.state_ranges[2][0]
        if self.pinball.ball.ydot > self.state_ranges[2][1]:
            self.pinball.ball.ydot = self.state_ranges[2][1]
        if self.pinball.ball.ydot < self.state_ranges[2][0]:
            self.pinball.ball.ydot = self.state_ranges[2][0]

    def collision_point(self, x, y):
        ballpoint = shapely.geometry.Point((x,y))
        ballcircle = ballpoint.buffer(0.03)
        for obs in self.pinball.obstacles:
            if obs.shapely_obs.contains(ballcircle) or ballcircle.intersects(obs.shapely_obs):
                return True
        return False

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)


###################### render functions ############################

    def render(self, mode='human'):
        if self.interactive:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.environment_view.blit(self.pinball.ball.position)
            pygame.display.flip()

    def close(self):
        if self.interactive:
            pygame.quit()

class PinballView:
    """ This class displays a :class:`PinballModel`

    This class is used in conjunction with the :func:`run_pinballview`
    function, acting as a *controller*.

    We use `pygame <http://www.pygame.org/>` to draw the environment.

    """
    def __init__(self, screen, model):
        """
        :param screen: a pygame surface
        :type screen: :class:`pygame.Surface`
        :param model: an instance of a :class:`PinballModel`
        :type model: :class:`PinballModel`
        """
        self.screen = screen
        self.model = model

        self.DARK_GRAY = [64, 64, 64]
        self.LIGHT_GRAY = [232, 232, 232]
        self.BALL_COLOR = [0, 0, 255]
        self.TARGET_COLOR = [255, 0, 0]

        # Draw the background
        # self.background_surface = pygame.Surface(screen.get_size())
        self.screen.fill(self.LIGHT_GRAY)
        for obs in self.model.obstacles:
            points = [self._to_pixels(o) for o in obs.points]
            # points = map(self._to_pixels, obs.points)
            pygame.draw.polygon(self.screen, self.DARK_GRAY, points, 0)

        pygame.draw.circle(
            self.screen, self.TARGET_COLOR, self._to_pixels(self.model.target_pos), int(self.model.target_rad*self.screen.get_width()))
        pygame.display.flip()

    def _to_pixels(self, pt):
        """ Converts from real units in the 0-1 range to pixel units

        :param pt: a point in real units
        :type pt: list
        :returns: the input point in pixel units
        :rtype: list

        """
        return [int(pt[0] * self.screen.get_width()), int(pt[1] * self.screen.get_height())]

    def blit(self, position):
        """ Blit the ball onto the background surface """
        # self.screen.blit(self.background_surface, (0, 0))
        # pygame.draw.circle(self.screen, self.BALL_COLOR,
        #                    self._to_pixels(self.model.ball.position), int(self.model.ball.radius*self.screen.get_width()))

        self.screen.fill(self.LIGHT_GRAY)
        for obs in self.model.obstacles:
            points = [self._to_pixels(o) for o in obs.points]
            pygame.draw.polygon(self.screen, self.DARK_GRAY, points, 0)

        pygame.draw.circle(
            self.screen, self.TARGET_COLOR, self._to_pixels(self.model.target_pos), int(self.model.target_rad*self.screen.get_width()))
        
        pygame.draw.circle(
            self.screen, self.BALL_COLOR, self._to_pixels(position), int(self.model.ball.radius*self.screen.get_width()))
        
        pygame.display.flip()