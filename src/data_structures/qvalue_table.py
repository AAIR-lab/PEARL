class Qtable:
    def __init__(self):
        self._qtable = {}

    ##################### set qvalues #####################
    def initialize_qvalue(self, state, action, q_init):
        if state not in self._qtable:
            self._qtable[state] = {}
        if action not in self._qtable[state]:
            self._qtable[state][action] = q_init

    def update_qvalue(self, state, action, qvalue):
        if state not in self._qtable:
            self._qtable[state] = {}
        self._qtable[state][action] = qvalue

    def update_qvalues(self, state, qvalues):
        self._qtable[state] = qvalues
    
    def initialize_qtable(self, qtable_full):
        self._qtable = qtable_full

    ##################### get qvalues #####################
    def get_qvalue(self, state, action, initial_value):
        if state not in self._qtable or action not in self._qtable[state]:
            self.initialize_qvalue(state, action, initial_value)
        return self._qtable[state][action]

    def get_qvalues(self, state):
        if state not in self._qtable:
            return {}
        return self._qtable[state]

    def get_max_qvalue(self, state, initial_value=0.0):
        if state not in self._qtable or len(self._qtable[state]) == 0:
            return initial_value 
        return max(self._qtable[state].values())

    ##################### get actions #####################
    def get_actions(self, state):
        if state not in self._qtable:
            return []
        return list(self._qtable[state].keys())

    def get_best_action(self, state, random_generator):
        actions = self.get_best_actions(state)
        if len(actions) > 0:
            action = random_generator.choice(actions)
        else:
            action = None
            print("No actions available to choose from.")
        return action

    def get_best_actions(self, state):
        if state not in self._qtable or len(self._qtable[state]) == 0:
            return []
        max_value = max(self._qtable[state].values())
        actions = []
        for action, qvalue in self._qtable[state].items():
            if qvalue == max_value:
                actions.append(action)
        return actions

        
