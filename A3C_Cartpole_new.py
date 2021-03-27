import matplotlib.pyplot as plt
import multiprocessing as mp
import tensorflow as tf
import numpy as np
from datetime import datetime
import time
import gym

UNIT_ACTOR_1 = 128  # number of units in 1st Actor layer
UNIT_ACTOR_2 = 64  # number of units in 2nd Actor layer
UNIT_CRITIC_1 = 128  # number of units in 1st Critic layer
UNIT_CRITIC_2 = 64  # number of units in 2nd Critic layer

EPISODE_MAX = 10000  # max episode of each local agent
STEP_MAX = 5  # max step before update network

GAMMA = 0.99  # reward discount
BETA = 0.01  # exploration coefficient
LR = 0.00005  # learning rate

class NeuralNetwork:
    def __init__(self, ob_shape, action_shape):
        """
        Create Actor-Critic network.

        :param ob_shape: observation dimension
        :param action_shape: action dimension
        """
        # Actor Network
        inputs = tf.keras.Input(shape=(ob_shape,))
        x = tf.keras.layers.Dense(UNIT_ACTOR_1, activation='relu')(inputs)
        x = tf.keras.layers.Dense(UNIT_ACTOR_2, activation='relu')(x)
        outputs_a = tf.keras.layers.Dense(action_shape)(x)

        self.model_Actor = tf.keras.Model(inputs=inputs,
                                          outputs=outputs_a,
                                          name='Actor_Net')

        # Critic Network
        y= tf.keras.layers.Dense(UNIT_CRITIC_1, activation='relu')(inputs)
        y= tf.keras.layers.Dense(UNIT_CRITIC_2, activation='relu')(y)
        outputs_c = tf.keras.layers.Dense(1)(y)

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
    def __init__(self, ob_shape, action_shape):
        super().__init__(ob_shape, action_shape)

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
    def __init__(self, ob_shape, action_shape, ini_weight, seed, index, plot=False):
        """
        :param ob_shape: observation dimension
        :param action_shape: action dimension
        :param ini_weight: initial weights (the same as the global network)
        :param seed: env random seed
        :param index: agent index
        :param plot: plot the test reward
        """
        super().__init__(ob_shape, action_shape)

        # initialize the weights
        self.model_ActorCritic.set_weights(weights=ini_weight)

        # create game
        self.env = gym.make('CartPole-v0')
        self.env.seed(seed)

        self.index = index

        self.plot = plot
        self.test_rewards = []

    def train(self):
        print('Agent', self.index, 'is training.')

        # reset the game
        state = self.env.reset()
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

                    # Probe
                    # if step % 20 == 0:
                    #     print(policy.numpy()[0])

                    # Perform action
                    action = np.random.choice(2, size=1, p=policy.numpy().reshape(-1))[0]
                    state, reward, done, _ = self.env.step(action)
                    reward = self.custom_reward(state)
                    #self.env.render()

                    # Extract the selected log_policy
                    log_policy = tf.reduce_sum(tf.reshape(tf.one_hot(action, 2), (1, -1)) * log_policy)

                    rewards.append(reward)
                    policys.append(policy)
                    log_policys.append(log_policy)
                    entropies.append(entropy)
                    values.append(value)

                    if done:
                        state = self.env.reset()
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
                loss = - actor_loss + critic_loss + BETA * entropy_loss

            # Compute gradient
            d = t.gradient(loss, self.model_ActorCritic.trainable_weights)

            lock.acquire()
            local_end.send(d)
            params = local_end.recv()
            self.model_ActorCritic.set_weights(params)
            lock.release()

            # Test per 100 updates
            if i_episode % 100 == 0:
                print('Agent', self.index, 'is testing.')
                total_rewards = []
                for round in range(5):
                    state = self.env.reset()
                    total_reward = 0
                    while 1:
                        logits = self.model_Actor(state.reshape((1, -1)))
                        policy = tf.nn.softmax(logits)
                        action = np.argmax(policy)
                        state, reward, done, _ = self.env.step(action)
                        reward = self.custom_reward(state)
                        self.env.render()
                        total_reward += reward
                        if done:
                            state = self.env.reset()
                            total_rewards.append(total_reward)
                            break
                test_reward_ave = np.mean(total_rewards)
                print('Test total rewards (averaged):', test_reward_ave)

                self.test_rewards.append(test_reward_ave)

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



    def custom_reward(self, state):
        x, x_dot, theta, theta_dot = state
        r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
        r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
        return r1 + r2


def local_run(index, seed, plot=False):
    local_agent = LocalAgent(ob_shape=4,
                             action_shape=2,
                             ini_weight=ini_weights,
                             seed=seed,
                             index=index,
                             plot=plot)
    local_agent.train()




if __name__ == '__main__':
    """
    ------------------------A3C-----------------------
    """
    # Create global network
    global_net = GlobalNetwork(ob_shape=4, action_shape=2)
    ini_weights = global_net.get_weights()

    # multiprocessing
    lock = mp.Lock()
    center_end, local_end = mp.Pipe()
    p1 = mp.Process(target=local_run, args=(1, 100, True))
    p2 = mp.Process(target=local_run, args=(2, 200))
    p3 = mp.Process(target=local_run, args=(3, 300))
    p4 = mp.Process(target=local_run, args=(4, 400))

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    global_net.receive_grad(4)


