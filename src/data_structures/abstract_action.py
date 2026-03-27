class AbstractAction(object):


    def __init__(self, action, discrete_action):
        self.action_params = tuple([tuple(param_range) for param_range in action])
        self.discrete_action = discrete_action

    def __hash__(self) -> int:
        return hash(self.action_params) + hash(self.discrete_action)

    def __str__(self) -> str:
        return  str(self.discrete_action) + "_" + str(self.action_params)

    def __repr__(self) -> str:
        return str(self.discrete_action) + "_" + str(self.action_params)

    def __eq__(self, other):
        return self.discrete_action == other.discrete_action and self.action_params == other.action_params

    