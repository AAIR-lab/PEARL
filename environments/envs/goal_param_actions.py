import gym
import gym_goal
import numpy as np
from src.misc import utils
from gym_goal.envs.config import GOAL_WIDTH, PITCH_LENGTH, PITCH_WIDTH

np.bool = np.bool_

"""
Continuous state variables: 
SCALE_VECTOR = np.array([PITCH_LENGTH / 2, PITCH_WIDTH, 2.0, 2.0, 2 * np.pi,
                         PITCH_LENGTH / 2, PITCH_WIDTH, 2.0, 2.0, 2 * np.pi,
                         PITCH_LENGTH / 2, PITCH_WIDTH, 6.0, 6.0])
SHIFT_VECTOR = np.array([0.0, PITCH_WIDTH / 2, 1.0, 1.0, np.pi,
                         0.0, PITCH_WIDTH / 2, 1.0, 1.0, np.pi,
                         0.0, PITCH_WIDTH / 2, 3, 3])
LOW_VECTOR = -SHIFT_VECTOR
HIGH_VECTOR = np.array(SCALE_VECTOR-SHIFT_VECTOR)

state = np.concatenate((
    self.player.position,
    self.player.velocity,
    [self.player.orientation],
    self.goalie.position,
    self.goalie.velocity,
    [self.goalie.orientation],
    self.ball.position,
    self.ball.velocity))

self.player.position[0] = state[0]
self.player.position[1] = state[1]
self.player.velocity[0] = state[2]
self.player.velocity[1] = state[3]
self.player.orientation = state[4]
self.goalie.position[0] = state[5]
self.goalie.position[1] = state[6]
self.goalie.velocity[0] = state[7]
self.goalie.velocity[1] = state[8]
self.goalie.orientation = state[9]
self.ball.position[0] = state[10]
self.ball.position[1] = state[11]
self.ball.velocity[0] = state[12]
self.ball.velocity[1] = state[13]


Parameterized actions: 
ACTION_LOOKUP = {
    0: KICK_TO,
    1: SHOOT_GOAL,
    2: SHOOT_GOAL,
}
# field bounds seem to be 0, PITCH_LENGTH / 2, -PITCH_WIDTH / 2, PITCH_WIDTH / 2
PARAMETERS_MIN = [
    np.array([0, -PITCH_WIDTH / 2]),  # -15
    np.array([-GOAL_WIDTH / 2]),  # -7.01
    np.array([0]),  # 0 #-7.01
]
PARAMETERS_MAX = [
    np.array([PITCH_LENGTH, PITCH_WIDTH / 2]),  # 40, 15
    np.array([0]),  # 0 #7.01
    np.array([GOAL_WIDTH / 2]),  # 7.01
]
"""


class GoalWrapperEnv(gym.Wrapper):
    def __init__(self, interactive):
        env = gym.make('Goal-v0')
        self.interactive = interactive
        env = GoalObservationWrapper(env)
        super(GoalWrapperEnv, self).__init__(env)

        self.state_ranges = self.get_state_ranges()
        self.action_param_ranges = self.get_action_param_ranges()

        self.fixed_values = [] # used for plotting abstractions
        self.initial_refinement_order = [[1,1,1,1,1] + [0,0,0,0,0] + [1,1,1,1] + [0,0,0]]
        self.use_mean_action = True
        
    def initialize_problem(self, step_max):
        state = self.reset()
        self.step_max = step_max

    def reset(self):
        self.steps = 0
        # state, _ = super().reset()
        state = super().reset()
        return state

    def get_state_ranges(self):
        state_ranges = []
        self.is_int_state_variable = []
        for i in range(len(self.observation_space.spaces[0].low)):
            low = utils.avoid_negative_zero(np.float32(self.observation_space.spaces[0].low[i]))
            high = utils.avoid_negative_zero(np.float32(self.observation_space.spaces[0].high[i]))
            state_ranges.append([low, high])
            self.is_int_state_variable.append(False)
        return state_ranges

    def get_action_param_ranges(self):
        self.actions = ['run', 'hop', 'leap']
        self.action_size = len(self.actions)
        self.is_action_space_discrete = False
        action_param_ranges = {}
        for i, action_space in enumerate(self.action_space.spaces[1].spaces):
            ranges = []
            length = int(len(action_space.low))
            for j in range(0,length):
                ranges.append([action_space.low[j], action_space.high[j]])
            action_param_ranges[i] = ranges

        # old_as = self.action_space
        # num_actions = old_as.spaces[0].n
        # self.action_space = gym.spaces.Tuple((
        #     old_as.spaces[0],  # actions
        #     *(gym.spaces.Box(old_as.spaces[1].spaces[i].low, old_as.spaces[1].spaces[i].high, dtype=np.float32)
        #       for i in range(0, num_actions))
        # ))

        self.is_int_discrete_action_params = {}
        for i in range(0, self.action_size):
            self.is_int_discrete_action_params[i] = False 

        # print("action_param_ranges: ", action_param_ranges)
        return action_param_ranges
    
    def pad_action(self, act, act_param):
        params = [np.zeros((1,)), np.zeros((1,)), np.zeros((1,))]
        params[act] = act_param
        return (act, params)
    
    def step(self, action):
        padded_action = self.pad_action(action[0], action[1])
        state_steps, reward, end_episode, info = super().step(padded_action)
        self.steps += 1
        info["steps"] = self.steps
        return state_steps[0], reward, end_episode, info

    def render(self, mode='human'):
        if self.interactive:
            super().render()



class GoalObservationWrapper(gym.ObservationWrapper):
    """
    Extends the Goal domain state with keeper and ball difference features.
    """

    def __init__(self, env):
        super(GoalObservationWrapper, self).__init__(env)
        base_state = env.get_state()
        ball_feats = self.ball_features(base_state)
        keeper_feats = self.keeper_features(base_state)
        newshape = (base_state.shape[0] + ball_feats.shape[0] + keeper_feats.shape[0],)
        low = np.zeros(newshape)
        low[:14] = env.observation_space.spaces[0].low
        # since keeper-ball difference vector is normalised
        low[14] = -1.
        low[15] = -1.
        low[16] = -GOAL_WIDTH / 2
        high = np.ones(newshape)
        high[:14] = env.observation_space.spaces[0].high
        # since keeper-ball difference vector is normalised
        high[14] = 1.
        high[15] = 1.
        high[16] = GOAL_WIDTH
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=low, high=high, dtype=np.float32),
            gym.spaces.Discrete(200),  # steps (200 limit is an estimate)
        ))

    @staticmethod
    def keeper_projection(state):
        if state[5] == state[10]:
            if state[6] < state[11]:
                return -GOAL_WIDTH / 2
            else:
                return GOAL_WIDTH / 2
        grad = (state[6] - state[11]) / (state[5] - state[10])
        y_int = state[11] - grad * state[10]
        pos = grad * PITCH_LENGTH / 2 + y_int
        return np.clip(pos, -GOAL_WIDTH / 2, GOAL_WIDTH)

    def keeper_features(self, state):
        """
        Returns [g], where g is the projection
        of the goalie onto the goal line.
        """
        _state = state
        yval = self.keeper_projection(_state)
        return np.array([yval])

    @staticmethod
    def position_features(state):
        """
        Returns [1 p p^2], containing the squared features
        of the player position.
        """
        xval = state[0] / (PITCH_LENGTH / 2)
        yval = state[1] / (PITCH_WIDTH / 2)
        return np.array([1., xval, yval, xval ** 2, yval ** 2])

    def ball_features(self, state):
        """ Returns ball-based position features. """
        ball = np.array((state[10], state[11]))
        keeper = np.array((state[5], state[6]))
        diff = (ball - keeper) / np.linalg.norm(ball - keeper)
        return np.array([diff[0], diff[1]])

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self.observation(obs)
        return obs
    
    def observation(self, obs):
        # state, steps = obs
        state = obs
        state = np.concatenate((state, self.ball_features(state), self.keeper_features(state)))
        # return (state, steps)
        return state

    def step(self, action):
        state_steps, reward, end_episode, info = self.env.step(action)
        state = self.observation(state_steps[0])
        state_steps = (state, state_steps[1])
        return state_steps, reward, end_episode, info