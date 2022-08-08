import numpy as np

class SingleReach:
    def __init__(self, type):
        if type == 'fix':
            self.target_pos = np.array([0.5,0.6])
        if type == 'random':
            self.target_pos = (2*np.random.rand(2,1)) - 1

        self.target_size = 0.02
        self.dwell_time = 20
        self.hand_in = False
        self.trial_state = 0    # State of trial:  -1 loss, 0 still going, 1 Finished
        self.dwell_counter = 0
        self.screen = self.target_pos
        
    def get_reward(self, arm_pos):
        # Hand out of the target
        if np.linalg.norm(arm_pos-self.target_pos) >= self.target_size:
            if self.hand_in == True:
                self.reward = -10
                self.trial_state = -1
            else:
                self.reward = - np.linalg.norm(arm_pos-self.target_pos)
        # Hand in the target
        else:
            # First time
            if self.hand_in == False:
                self.reward = 0
                self.hand_in = True
                self.dwell_counter += 1
            # Has been in before
            else:
                if self.dwell_counter < self.dwell_time:
                    self.reward = 0
                    self.dwell_counter += 1
                else:
                    self.reward = 10
                    self.trial_state = 1
                    self.dwell_counter += 1
        return self.reward, self.target_pos, self.trial_state