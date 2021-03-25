import matplotlib.pyplot as plt
import multiprocessing as mp
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from datetime import datetime
import time
import gym

import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game

UNIT_ACTOR_1 = 128  # number of units in 1st Actor layer
UNIT_ACTOR_2 = 64  # number of units in 2nd Actor layer
UNIT_CRITIC_1 = 128  # number of units in 1st Critic layer
UNIT_CRITIC_2 = 64  # number of units in 2nd Critic layer

EPISODE_MAX = 1000  # max episode of each local agent
STEP_MAX = 10  # max step before update network

GAMMA = 0.99  # reward discount
BETA = 0.01  # exploration coefficient
LR = 0.0001  # learning rate

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial).initialized_value()

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial).initialized_value()

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

class NeuralNetwork:
    def __init__(self, action_shape):
        """
        Create Actor-Critic network.

        :param ob_shape: observation dimension
        :param action_shape: action dimension
        """
        
        # Actor Network
        # network weights
        W_conv1_a = weight_variable([8, 8, 1, 32])
        b_conv1_a = bias_variable([32])

        W_conv2_a = weight_variable([4, 4, 32, 64])
        b_conv2_a = bias_variable([64])

        W_conv3_a = weight_variable([3, 3, 64, 64])
        b_conv3_a = bias_variable([64])

        W_fc1_a = weight_variable([1600, 512])
        b_fc1_a = bias_variable([512])

        W_fc2_a = weight_variable([512, action_shape])
        b_fc2_a = bias_variable([action_shape])
        # input layer
        
        inputs = tf.keras.Input(shape=(80,80,1))
        # hidden layers
        h_conv1_a = tf.nn.relu(conv2d(inputs, W_conv1_a, 4) + b_conv1_a)
        h_pool1_a = max_pool_2x2(h_conv1_a)

        h_conv2_a = tf.nn.relu(conv2d(h_pool1_a, W_conv2_a, 2) + b_conv2_a)

        h_conv3_a = tf.nn.relu(conv2d(h_conv2_a, W_conv3_a, 1) + b_conv3_a)

        h_conv3_flat_a = tf.reshape(h_conv3_a, [-1, 1600])

        h_fc1_a = tf.nn.relu(tf.matmul(h_conv3_flat_a, W_fc1_a) + b_fc1_a)

        #out layer
        outputs_a = tf.matmul(h_fc1_a, W_fc2_a) + b_fc2_a



        self.model_Actor = tf.keras.Model(inputs=inputs,
                                          outputs=outputs_a,
                                          name='Actor_Net')

        # Critic Network
        # network weights
        W_conv1_c = weight_variable([8, 8, 1, 32])
        b_conv1_c = bias_variable([32])

        W_conv2_c = weight_variable([4, 4, 32, 64])
        b_conv2_c = bias_variable([64])

        W_conv3_c = weight_variable([3, 3, 64, 64])
        b_conv3_c = bias_variable([64])

        W_fc1_c = weight_variable([1600, 512])
        b_fc1_c = bias_variable([512])

        W_fc2_c = weight_variable([512, 1])
        b_fc2_c = bias_variable([1])


        # hidden layers
        h_conv1_c = tf.nn.relu(conv2d(inputs, W_conv1_c, 4) + b_conv1_c)
        h_pool1_c = max_pool_2x2(h_conv1_c)

        h_conv2_c = tf.nn.relu(conv2d(h_pool1_c, W_conv2_c, 2) + b_conv2_c)


        h_conv3_c = tf.nn.relu(conv2d(h_conv2_c, W_conv3_c, 1) + b_conv3_c)

        h_conv3_flat_c = tf.reshape(h_conv3_c, [-1, 1600])

        h_fc1_c = tf.nn.relu(tf.matmul(h_conv3_flat_c, W_fc1_c) + b_fc1_c)

        # readout layer
        outputs_c = tf.matmul(h_fc1_c, W_fc2_c) + b_fc2_c


        self.model_Critic = tf.keras.Model(inputs=inputs,
                                           outputs=outputs_c,
                                           name='Critic_Net')

        # Actor-Critic Network
        self.model_ActorCritic = tf.keras.Model(inputs=inputs,
                                                outputs=[outputs_a, outputs_c],
                                                name='ActorCritic_Net')

    def show_model(self):
        """
        Show the structure of the neural network.
        """
        self.model_Actor.summary()
        self.model_Critic.summary()
        self.model_ActorCritic.summary()

        # generate .PNG figure
        tf.keras.utils.plot_model(self.model_Actor, 'Actor_Network.png', show_shapes=True)
        tf.keras.utils.plot_model(self.model_Critic, 'Critic_Network.png', show_shapes=True)
        tf.keras.utils.plot_model(self.model_ActorCritic, 'Actor_Critic_Network.png', show_shapes=True)


class GlobalNetwork(NeuralNetwork):
    """
    Global network.
    Receive gradients from local networks, updates, send back new models.
    """
    def __init__(self, action_shape):
        super().__init__(action_shape)

        # Optimizer
        self.opti = tf.keras.optimizers.RMSprop(learning_rate=LR)

    def get_weights(self):
        """
        Get weights of the Actor-critic network.
        :return: weights
        """
        return self.model_ActorCritic.get_weights()

    def get_optimizer(self):
        """
        Get the optimizer.
        :return: optimizer
        """
        return self.opti

    def receive_grad(self, n_agents):
        """
        Receive gradients from local agents
        :param n_agents: number of local agents
        """
        done_counter = 0  # count how many agents are done
        while 1:
            rec = center_end.recv()  # receive message from local agents
            if rec == 'Done':
                done_counter += 1
                if done_counter == n_agents:
                    print('Training done!')
                    break
            else:
                d = rec
                self.opti.apply_gradients(zip(d, self.model_ActorCritic.trainable_weights))
                center_end.send(self.get_weights())


class LocalAgent(NeuralNetwork):
    """
    Local agent.
    Interact with the env and compute the gradient.
    Then send the gradient to the global network.
    """
    def __init__(self, action_shape, ini_weight, index, plot=False):
        """
        :param action_shape: action dimension
        :param ini_weight: initial weights (the same as the global network)
        :param index: agent index
        :param plot: plot the test reward
        """
        super().__init__(action_shape)

        # initialize the weights
        self.model_ActorCritic.set_weights(weights=ini_weight)

        # create game
        self.env = game.GameState()

        self.index = index

        self.plot = plot
        self.test_rewards = []

    def train(self):
        print('Agent', self.index, 'is training.')

        # reset the game
        step_counter = 0

        for i_episode in range(EPISODE_MAX):
            policys = []
            log_policys = []
            entropies = []
            values = []
            rewards = []

            with tf.GradientTape() as t:
                for step in range(STEP_MAX):
                    logits, value = self.model_ActorCritic(state.reshape((1, -1)))
                    policy = tf.nn.softmax(logits)
                    log_policy = tf.nn.log_softmax(logits)
                    entropy = tf.reduce_sum(policy * log_policy, keepdims=True)


                    # Perform action
                    action = np.random.choice(2, size=1, p=policy.numpy().reshape(-1))[0]

                    state_colored, reward, done = self.env.frame_step(action)
                    state = cv2.cvtColor(cv2.resize(state_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
                    _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
                    # Extract the selected log_policy
                    log_policy = tf.reduce_sum(tf.reshape(tf.one_hot(action, 2), (1, -1)) * log_policy)

                    rewards.append(reward)
                    policys.append(policy)
                    log_policys.append(log_policy)
                    entropies.append(entropy)
                    values.append(value)

                    if done:
                        #print('Max step:', step + 1)
                        break
                if done:
                    R = 0
                else:
                    R = value.numpy()[0, 0]
                actor_loss = 0
                critic_loss = 0
                entropy_loss = 0
                for i in range(step, -1, -1):
                    R = GAMMA * R + rewards[i]
                    actor_loss += (R - values[i]) * log_policys[i]
                    critic_loss += (R - values[i]) ** 2
                    entropy_loss += entropies[i]
                loss = - actor_loss + critic_loss - BETA * entropy_loss

            # Compute gradient
            d = t.gradient(loss, self.model_ActorCritic.trainable_weights)

            lock.acquire()
            local_end.send(d)
            params = local_end.recv()
            self.model_ActorCritic.set_weights(params)
            lock.release()

            # Test per 100 updates
            # if i_episode % 100 == 0:
            #     print('Agent', self.index, 'is testing.')
            #     total_rewards = []
            #     for round in range(5):
            #         state = self.env.reset()
            #         total_reward = 0
            #         while 1:
            #             logits = self.model_Actor(state.reshape((1, -1)))
            #             policy = tf.nn.softmax(logits)
            #             action = np.argmax(policy)
            #             state, reward, done, _ = self.env.step(action)
            #             reward = self.custom_reward(state)
            #             self.env.render()
            #             total_reward += reward
            #             if done:
            #                 state = self.env.reset()
            #                 total_rewards.append(total_reward)
            #                 break
            #     test_reward_ave = np.mean(total_rewards)
            #     print('Test total rewards (averaged):', test_reward_ave)

            #     self.test_rewards.append(test_reward_ave)

        lock.acquire()
        local_end.send('Done')
        lock.release()

        if self.plot:
            date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            np.save('logs/'+date, np.array(self.test_rewards))
            plt.plot(np.array(range(len(self.test_rewards))) * 100, self.test_rewards)
            plt.xlabel('Number of updates')
            plt.ylabel('Test rewards')
            plt.show()


def local_run(index, plot=False):
    local_agent = LocalAgent(action_shape=2,
                             ini_weight=ini_weights,
                             index=index,
                             plot=plot)
    local_agent.train()




if __name__ == '__main__':
    """
    ------------------------A3C-----------------------
    """
    # Create global network
    global_net = GlobalNetwork(action_shape=2)
    ini_weights = global_net.get_weights()

    # multiprocessing
    lock = mp.Lock()
    center_end, local_end = mp.Pipe()
    p1 = mp.Process(target=local_run, args=(1, True))
    p2 = mp.Process(target=local_run, args=(2, ))
    p3 = mp.Process(target=local_run, args=(3, ))
    p4 = mp.Process(target=local_run, args=(4, ))

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    global_net.receive_grad(4)


