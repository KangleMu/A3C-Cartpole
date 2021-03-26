import matplotlib.pyplot as plt
import multiprocessing as mp
import tensorflow as tf
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

EPISODE_MAX = 60000  # max episode of each local agent
STEP_MAX = 5  # max step before update network

GAMMA = 0.99  # reward discount
BETA = 0.1  # exploration coefficient
LR = 0.0001  # learning rate


class NeuralNetwork:
    def __init__(self, action_shape):
        """
        Create Actor-Critic network.
        :param action_shape: action dimension
        """
        
        # Shared network
        inputs = tf.keras.Input(shape=(80, 80, 4))
        x = tf.keras.layers.LayerNormalization(axis=1)(inputs)
        x = tf.keras.layers.Conv2D(32, 8, 4)(x)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, 4, 2)(x)
        x = tf.keras.layers.MaxPool2D((2, 2), 1)(x)
        x = tf.keras.layers.Flatten()(x)

        # Actor network
        x_a = tf.keras.layers.Dense(256, activation='relu')(x)
        outputs_a = tf.keras.layers.Dense(2)(x_a)

        # Critic network
        x_c = tf.keras.layers.Dense(256, activation='relu')(x)
        outputs_c = tf.keras.layers.Dense(1)(x_c)

        self.model_Actor = tf.keras.Model(inputs=inputs,
                                          outputs=outputs_a,
                                          name='Actor_Net')

        self.model_Critic = tf.keras.Model(inputs=inputs,
                                           outputs=outputs_c,
                                           name='Critic_Net')

        # Actor-Critic Network
        self.model_ActorCritic = tf.keras.Model(inputs=inputs,
                                                outputs=[outputs_a, outputs_c, x],
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
            elif rec == 'Test':
                center_end.send(self.get_weights())
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

        self.total_step = 0
        self.step_terminal = []

        self.index_to_save = 0

    def train(self, local_end, lock):
        print('Agent', self.index, 'is training.')

        # reset the game
        DO_NOTHING = np.array([0, 0])
        DO_NOTHING[np.random.randint(2)] = 1
        state_colored, reward, done = self.env.frame_step(DO_NOTHING)
        # convert states
        state = cv2.cvtColor(cv2.resize(state_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
        state = np.stack((state, state, state, state), axis=2)

        step_counter = 0

        for i_episode in range(EPISODE_MAX):
            policys = []
            log_policys = []
            entropies = []
            values = []
            rewards = []

            with tf.GradientTape() as t:
                for step in range(STEP_MAX):

                    self.total_step += 1

                    state_reshape = tf.cast(state.reshape((1, 80, 80, 4)), tf.float32)

                    logits, value, test = self.model_ActorCritic(state_reshape)
                    policy = tf.nn.softmax(logits)
                    log_policy = tf.nn.log_softmax(logits)
                    entropy = tf.reduce_sum(policy * log_policy, keepdims=True)
                    print(policy.numpy())

                    # perform action
                    action = np.random.choice(2, size=1, p=policy.numpy().reshape(-1))[0]
                    # reshape action
                    action_vec = np.zeros(2)
                    action_vec[action] = 1

                    state_colored_next, reward, done = self.env.frame_step(action_vec)
                    # convert states
                    state_next = cv2.cvtColor(cv2.resize(state_colored_next, (80, 80)), cv2.COLOR_BGR2GRAY)
                    _, state_next = cv2.threshold(state_next, 1, 255, cv2.THRESH_BINARY)
                    state_next = np.reshape(state_next, (80, 80, 1))
                    state = np.append(state_next, state[:, :, :3], axis=2)

                    # Extract the selected log_policy
                    log_policy = tf.reduce_sum(tf.reshape(tf.one_hot(action, 2), (1, -1)) * log_policy)

                    rewards.append(reward)
                    policys.append(policy)
                    log_policys.append(log_policy)
                    entropies.append(entropy)
                    values.append(value)

                    if done:
                        #print('Max step:', step + 1)
                        self.step_terminal.append(self.total_step)
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
                loss = - actor_loss + critic_loss + BETA * entropy_loss

            # Compute gradient
            d = t.gradient(loss, self.model_ActorCritic.trainable_weights)

            lock.acquire()
            local_end.send(d)
            params = local_end.recv()
            self.model_ActorCritic.set_weights(params)


            print("Agent {0} is fininsh the {1}th episode".format(self.index, i_episode))
            if self.total_step > 5000 * self.index_to_save:
                self.index_to_save += 1
                np.save("log_socre_" + str(self.index), np.array(self.step_terminal))

            lock.release()

        lock.acquire()
        local_end.send('Done')
        lock.release()

class TestAgent(NeuralNetwork):
    """
    Test agent.
    """
    def __init__(self, action_shape, ini_weight, plot=False):
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

        self.plot = plot
        self.test_rewards = []

        self.total_step = 0
        self.step_terminal = []

        self.index_to_save = 0

    def test(self, local_end, lock):

        # reset the game
        DO_NOTHING = np.array([0, 0])
        DO_NOTHING[np.random.randint(2)] = 1
        state_colored, reward, done = self.env.frame_step(DO_NOTHING)
        # convert states
        state = cv2.cvtColor(cv2.resize(state_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
        state = np.stack((state, state, state, state), axis=2)

        step_counter = 0

        while 1:
            rewards = []

            for round in range(5):
                while 1:
                    layer = tf.keras.layers.LayerNormalization(axis=1)
                    state_reshape = tf.cast(state.reshape((1, 80, 80, 4)), tf.float32)
                    normaliz_state = layer(state_reshape)

                    logits, value, test = self.model_ActorCritic(normaliz_state)
                    policy = tf.nn.softmax(logits)
                    action = np.argmax(policy)

                    # reshape action
                    action_vec = np.zeros(2)
                    action_vec[action] = 1

                    state_colored_next, reward, done = self.env.frame_step(action_vec)
                    # convert states
                    state_next = cv2.cvtColor(cv2.resize(state_colored_next, (80, 80)), cv2.COLOR_BGR2GRAY)
                    _, state_next = cv2.threshold(state_next, 1, 255, cv2.THRESH_BINARY)
                    state_next = np.reshape(state_next, (80, 80, 1))
                    state = np.append(state_next, state[:, :, :3], axis=2)

                    rewards.append(reward)

                    if done:
                        print('Test total rewards:', sum(rewards))
                        break

                lock.acquire()
                local_end.send('Test')
                params = local_end.recv()
                self.model_ActorCritic.set_weights(params)
                lock.release()

            time.sleep(5)


def local_run(index, ini_weights, local_end, lock, plot=False):
    local_agent = LocalAgent(action_shape=2,
                             ini_weight=ini_weights,
                             index=index,
                             plot=plot)
    local_agent.train(local_end, lock)


def test_run(ini_weights, local_end, lock, plot=False):
    test_agent = TestAgent(action_shape=2,
                           ini_weight=ini_weights,
                           plot=plot)
    test_agent.test(local_end, lock)


if __name__ == '__main__':
    """
    ------------------------A3C-----------------------
    """
    # Create global network
    global_net = GlobalNetwork(action_shape=2)
    ini_weights = global_net.get_weights()

    # multiprocessing
    mp.set_start_method('spawn')
    lock = mp.Lock()
    center_end, local_end = mp.Pipe()
    p1 = mp.Process(target=local_run, args=(1, ini_weights, local_end, lock, True))
    p2 = mp.Process(target=local_run, args=(2, ini_weights, local_end, lock))
    p3 = mp.Process(target=local_run, args=(3, ini_weights, local_end, lock))

    p_test = mp.Process(target=test_run, args=(ini_weights, local_end, lock))  # thread of test agent

    p1.start()
    p2.start()
    p3.start()

    p_test.start()

    global_net.receive_grad(4)


