
class Transition():
    def __init__(self, state, rounded_state, action, next_state, rounded_next_state, reward, done, success, state_abs, action_abs, next_state_abs, reward_abs, steps_in_abs_state):
        self.state = state
        self.rounded_state = rounded_state
        self.action = action
        self.next_state = next_state
        self.rounded_next_state = rounded_next_state
        self.reward = reward
        self.done = done
        self.success = success
        self.state_abs = state_abs
        self.action_abs = action_abs
        self.next_state_abs = next_state_abs
        self.reward_abs = reward_abs
        self.steps_in_abs_state = steps_in_abs_state
    
    def __str__(self) -> str:
        return f"{self.rounded_state}, {self.action}, {self.rounded_next_state}, {self.reward_abs}"


class Trace():
    def __init__(self):
        self.trace = []

    def append_transition(self, transition):
        self.trace.append(transition)

    def compress_trace(self, trace):
        compressed_trace = []
        count = 0
        for i in range(len(trace)):
            if i == 0 or i == len(trace)-1:
                trace[i].steps_in_abs_state = count+1
                compressed_trace.append(trace[i])
            else:
                if trace[i].state_abs == trace[i-1].next_state_abs:
                    count += 1
                else:
                    trace[i].steps_in_abs_state = count+1
                    compressed_trace.append(trace[i])
                    count = 0
        return compressed_trace

    def print_trace(self):
        # compressed_trace = self.compress_trace(self.trace)
        compressed_trace = self.trace
        for transition in compressed_trace:
            print(f"\t{transition.state_abs}, {transition.state}, {transition.action_abs}, {transition.next_state_abs}, {transition.reward_abs}, {transition.success}, {transition.steps_in_abs_state}")

        