import numpy as np
import tensorflow as tf
from time import time

class SeqEnv:
    def __init__(self, plant,visual_kernel, task, **kwargs):
        self.plant = plant
        self.task = task
        self.visual_kernel = visual_kernel
        self.plant.excitation_noise_sd=  kwargs.get('motor_noise_sd', 0)
        self.num_muscle = self.plant.n_muscles
        [self.joint_state, self.cart_state, self.muscle_state, self.geometry_state] = plant.get_initial_state(batch_size=1)

    def get_init_state(self):
        [self.joint_state, self.cart_state, self.muscle_state, self.geometry_state] = self.plant.get_initial_state()

        target_visual = self.visual_kernel.encode(self.task.target_pos[0].reshape(-1,1), self.task.target_pos[1].reshape(-1,1))
        hand_visual = self.visual_kernel.encode(self.cart_state.numpy()[0][0].reshape(-1,1), self.cart_state.numpy()[0][1].reshape(-1,1))
        self.visual_feedback = np.concatenate((hand_visual.reshape(self.visual_kernel.n_kernel,self.visual_kernel.n_kernel,1) , \
                                               target_visual.reshape(self.visual_kernel.n_kernel,self.visual_kernel.n_kernel,1)),axis=-1)
        return self.visual_feedback


    def step(self, muscle_input):
        [self.joint_state, self.cart_state, self.muscle_state, self.geometry_state] = \
        self.plant(tf.convert_to_tensor(muscle_input,dtype='float') ,self.joint_state, self.muscle_state, self.geometry_state)
        self.reward, self.target_pos, self.trial_state = self.task.get_reward(self.cart_state.numpy()[0][:2])  
        
        target_visual = self.visual_kernel.encode(self.target_pos[0].reshape(-1,1), self.target_pos[1].reshape(-1,1))
        hand_visual = self.visual_kernel.encode(self.cart_state.numpy()[0][0].reshape(-1,1), self.cart_state.numpy()[0][1].reshape(-1,1))
        self.visual_feedback = np.concatenate((hand_visual.reshape(self.visual_kernel.n_kernel,self.visual_kernel.n_kernel,1) , \
                                               target_visual.reshape(self.visual_kernel.n_kernel,self.visual_kernel.n_kernel,1)),axis=-1)
        return self.reward, self.visual_feedback, self.trial_state

class SeqEnvSimple:
    # This environemnt uses simple x-y coordinate for visual feedback and motornet as arm model.
    def __init__(self, plant, task, **kwargs):
        self.plant = plant
        self.task = task
        self.plant.excitation_noise_sd=  kwargs.get('motor_noise_sd', 0)
        self.num_muscle = self.plant.n_muscles
        [self.joint_state, self.cart_state, self.muscle_state, self.geometry_state] = plant.get_initial_state(batch_size=1)

    def get_init_state(self):
        [self.joint_state, self.cart_state, self.muscle_state, self.geometry_state] = self.plant.get_initial_state()

        target_visual = self.task.target_pos.reshape(-1,1)
        hand_visual = self.cart_state.numpy()[0][0:2].reshape(-1,1)
        self.visual_feedback = np.concatenate((hand_visual, target_visual), axis=1)

        return self.visual_feedback


    def step(self, muscle_input):
        [self.joint_state, self.cart_state, self.muscle_state, self.geometry_state] = \
        self.plant(tf.convert_to_tensor(muscle_input,dtype='float') ,self.joint_state, self.muscle_state, self.geometry_state)
        self.reward, self.target_pos, self.trial_state = self.task.get_reward(self.cart_state.numpy()[0][:2])  
        
        target_visual = self.task.target_pos.reshape(-1,1)
        hand_visual = self.cart_state.numpy()[0][0:2].reshape(-1,1)
        self.visual_feedback = np.concatenate((hand_visual, target_visual), axis=1)
        
        return self.reward, self.visual_feedback, self.trial_state



class SeqEnvSimplePointMass:
    # This environment has simple x-y visual feedback and a simple point mass as endpoint model.
    def __init__(self, task, **kwargs):
        self.task = task
        self.num_muscle = 4
        self.z0 = np.array([[0,0,0,0]]).T
        self.z = np.copy(self.z0)
        self.n_ministep = 1

        self.dt = 0.01
        self.Kv = 0.1

        self.A = np.array([[0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, -self.Kv, 0],
                    [0, 0, 0, -self.Kv]])

        self.B = np.array([[0, 0, 0, 0], 
             [0, 0, 0, 0],
             [-1, 1, -1, 1],
             [1, 1, -1, -1]]) * (np.sqrt(2)/2)

    def get_init_state(self):
        target_visual = self.task.target_pos.reshape(-1,1)
        hand_visual = self.z[0:2].reshape(-1,1)
        self.visual_feedback = np.concatenate((hand_visual, target_visual), axis=1)
        return self.visual_feedback

    def step(self, muscle_input):
        for _ in range(self.n_ministep):
            u = np.array(muscle_input).reshape(self.num_muscle, 1)
            self.z = self.z + (self.A@self.z + self.B@u) * self.dt

        self.reward, self.target_pos, self.trial_state = self.task.get_reward(self.z[:2])  
        
        target_visual = self.task.target_pos.reshape(-1,1)
        hand_visual = self.z[0:2].reshape(-1,1)
        self.visual_feedback = np.concatenate((hand_visual, target_visual), axis=1)
        
        return self.reward, self.visual_feedback, self.trial_state