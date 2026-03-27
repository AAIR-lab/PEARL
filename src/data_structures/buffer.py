class TDErrorBuffer:
    def __init__(self):
        self._buffer = {}

    def get_values(self, state, action):
        if state not in self._buffer or action not in self._buffer[state]:
            return []
        return self._buffer[state][action]

    def add(self, state, action, tderror):
        if state not in self._buffer:
            self._buffer[state] = {}
        if action not in self._buffer[state]:
            self._buffer[state][action] = []
        self._buffer[state][action].append(tderror)
        


class QvalueBuffer:
    def __init__(self):
        self._buffer = {}

    def get_values(self, state, action):
        if state not in self._buffer or action not in self._buffer[state]:
            return []
        return self._buffer[state][action]

    def add(self, state, action, value):
        if state not in self._buffer:
            self._buffer[state] = {}
        if action not in self._buffer[state]:
            self._buffer[state][action] = []
        self._buffer[state][action].append(value)
        


class TraceBuffer():
    def __init__(self):
        self.buffer = []
        self.max_size = 1000000
        self.current_size = 0

    def append(self, trace):
        if self.current_size + len(trace.trace) < self.max_size:
            self.buffer.append(trace)
            self.current_size += len(trace.trace)