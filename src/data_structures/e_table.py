class Etable:
    def __init__(self):
        self._etable = {}

    ##################### set eligibility #####################
    def initialize_value(self, state, action):
        if state not in self._etable:
            self._etable[state] = {}
        if action not in self._etable[state]:
            self._etable[state][action] = 0

    def update_value(self, state, action, eligibility):
        self.initialize_value(state, action)
        self._etable[state][action] = eligibility

    def update_values(self, state, values):
        self._etable[state] = values

    ##################### get eligibility #####################
    def get_value(self, state, action):
        self.initialize_value(state, action)
        return self._etable[state][action]
