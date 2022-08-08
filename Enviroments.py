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
