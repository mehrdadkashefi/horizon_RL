import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class HorizonEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.frame_skip = 1
        self.dt_step = 0.01 * self.frame_skip 
        self.num_targets = 1
        self.delay_range = (np.array([300, 1000])/1000/self.dt_step).astype('int')
        self.visual_delay = int(70/1000/self.dt_step)
        self.visual_feedback = [np.zeros(8,)]
        self.last_a = np.zeros((6 ,))
        self.max_dist_reward = 2
        self.reward_div = 10

        self.perio_delay = int(40/1000/self.dt_step)
        self.perio_feedback = [np.zeros(2,)]

        # Time played
        self.t = 0
        self.dwell_go_max = 70
        self.dwell_go = self.dwell_go_max
        self.dwell = 70
        self.num_reach = 2

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, "./arm26.xml", self.frame_skip)

        
        
    def step(self, a):
        done = False
        if self.t <= self.dwell_go:
            vec = self.get_body_com("fingertip") - self.get_body_com("target_h")
            reward_dist = 1 / (np.linalg.norm(vec) + self.reward_div)           
            reward = reward_dist 
            if np.linalg.norm(vec) > 0.06:
                done = True
                reward -= 50
                print('Terminated: Bad Dwell')
        else:
            self.data.qpos.flat[2:4] = (0,0)
            if self.target_reached == 0:
                vec = self.get_body_com("fingertip") - self.get_body_com("target_1")
                reward_dist = 1 / (np.linalg.norm(vec) + self.reward_div) 
                reward = reward_dist 
                if np.linalg.norm(vec) < 0.06:
                    if self.dwell_counter == 0:
                        self.target_reached += 1
                        self.dwell_counter = self.dwell 
                        # Update targets
                        self.data.qpos.flat[4:6] = self.goal[1]
                        self.data.qpos.flat[6:8] = (0,0)
                        reward += 100
                    else:
                        self.dwell_counter -= 1
                        reward += 5
                else:     
                    if self.dwell_counter != self.dwell:
                        print('Bad Dwell in target 1!')
                        self.dwell_counter = self.dwell
                        done = True

            elif self.target_reached == 1:
                vec = self.get_body_com("fingertip") - self.get_body_com("target_1")
                reward_dist = 1 / (np.linalg.norm(vec) + self.reward_div) 
                reward = reward_dist 
                assert reward_dist >= 0
                if np.linalg.norm(vec) < 0.06:
                    if self.dwell_counter == 0:
                        done = True
                        reward += 500
                        print(' Game won!')
                    else:
                        self.dwell_counter -= 1
                        reward += 5
                else:
                    if self.dwell_counter != self.dwell:
                        print('Bad Dwell in target 2!')
                        self.dwell_counter = self.dwell
                        done = True
        
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        # Update visual and perio feedback with delays
        self.visual_feedback.insert(0, ob[2:])
        self.perio_feedback.insert(0, ob[:2])
        del self.visual_feedback[-1]
        del self.perio_feedback[-1]
        ob = np.concatenate((self.perio_feedback[-1], self.visual_feedback[-1]))
        info = dict(reward_dist=reward_dist, total_time = self.t)
        self.t = self.t + 1
        return ob, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        self.t = 0
        self.target_reached = 0
        self.dwell_go = np.random.randint(0, self.dwell_go_max)
        self.dwell_counter = self.dwell
        self.goal = []
        # Read initial pos and vel
        qpos = self.init_qpos
        qvel = self.init_qvel

        for _ in range(self.num_reach):
            x = np.random.uniform(0.52,1.74, (1,))
            y = np.random.uniform(0.52,1.74, (1,))
            self.goal.append(self.joint2cart(x, y))

        # Set targets to zero
        for h in range(3):
            qpos[4+(2*h):6+(2*h)] = (0,0) 
        # Set the goal
        self.horizon = np.random.choice(2) + 1
        for h in range(self.horizon):
            qpos[4+(2*h):6+(2*h)] = self.goal[h].reshape(-1, )
        # Set initial hand postion
        x = np.random.uniform(0.52,1.74, (1,))
        y = np.random.uniform(0.52,1.74, (1,))
        qpos[:2] = np.concatenate((x, y))
        qpos[2:4] = self.joint2cart(x, y).reshape(-1, )
        # Load visual delay list
        self.visual_feedback = [qpos[2:] for _ in range(self.visual_delay)]
        self.perio_feedback = [qpos[:2] for _ in range(self.perio_delay)]

        # Set the env
        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        th_pos = self.get_body_com('target_h')
        t1_pos = self.get_body_com('target_1')
        t2_pos = self.get_body_com('target_2')
              # Return Finger tip postion, joint position, home target pos, T1 target pos
        return np.concatenate((self.data.qpos.flat[:2], self.get_body_com('fingertip')[:2], th_pos[:2], t1_pos[:2], t2_pos[:2]))
    
    def joint2cart(self, th1, th2):
        x = 0.5*np.cos(th1) + 0.5*np.cos(th1+th2)
        y = 0.5*np.sin(th1) + 0.5*np.sin(th1+th2) 
        return np.array([x, y])