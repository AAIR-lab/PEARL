from collections import deque

class AbstractState(object):

    def __init__(self, state, identifier):
        self.state = state
        self.state_tuple = None
        self.id = identifier

    def __hash__(self) -> int:
        return hash(self.id)

    def __str__(self) -> str:
        return str([round(item,2) for item in self.state]) + "_" + str(self.id)

    def __repr__(self) -> str:
        return str(self.state) + "_" + str(self.id)

    def __eq__(self, other):     
        if not isinstance(other, AbstractState):
            return NotImplemented   
        return self.id == other.id
