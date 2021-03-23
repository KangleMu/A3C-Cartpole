import multiprocessing as mp
import tensorflow as tf
import numpy as np
import datetime
import gym

UNIT_ACTOR_1 = 128  # number of units in 1st Actor layer
UNIT_ACTOR_2 = 64  # number of units in 2nd Actor layer
UNIT_CRITIC_1 = 128  # number of units in 1st Critic layer
UNIT_CRITIC_2 = 64  # number of units in 2nd Critic layer

EPISODE_MAX = 500  # max episode of each local agent
STEP_MAX = 5  # max step before update network

GAMMA = 0.99  # reward discount
BETA = -1  # exploration coefficient
LEARNINGRATE = 0.00065  # learning rate

class CenterAgent:
    def __init__(self):
        # create Actor-Critic network:
        # 1. create Actor network:
        inputs = tf.keras.Input(shape=(9,))
        x = tf.keras.layers.Dense(UNIT_ACTOR_1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)  # 1st layer
        x = tf.keras.layers.Dense(UNIT_ACTOR_2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # 2nd layer
        outputs_mu = tf.keras.layers.Dense(2, activation='tanh')(x)  # 2 means (corresponding to 2 actions) in 3rd layer (linear)
        outputs_sigma2 = tf.keras.layers.Dense(2, activation='softplus', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # 2 variance in 3rd layer (with square)

        self.model_A = tf.keras.Model(inputs=inputs,
                                      outputs=[outputs_mu, outputs_sigma2],
                                      name='Actor Network')

        # 2. create Critic network:
        y = tf.keras.layers.Dense(UNIT_CRITIC_1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)  # 1st layer
        y = tf.keras.layers.Dense(UNIT_CRITIC_2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(y)  # 2nd layer
        outputs_c = tf.keras.layers.Dense(1)(y)  # 3rd layer

        self.model_C = tf.keras.Model(inputs=inputs,
                                      outputs=outputs_c,
                                      name='Critic Network')

        # 3. create Actor-Critic network
        self.model_AC = tf.keras.Model(inputs=inputs,
                                       outputs=[outputs_mu, outputs_sigma2, outputs_c],
                                       name='Actor-Critic Network')

        # learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=LEARNINGRATE,
            decay_steps=500,
            decay_rate=1,
            staircase=False)

        # create optimizer:
        self.opti_A = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=0.99, momentum=0)
        self.opti_C = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=0.99, momentum=0)

        # show models:
        self.model_A.summary()
        self.model_C.summary()
        self.model_AC.summary()

        # plot models:
        # tf.keras.utils.plot_model(self.model_A, 'Actor Network.png', show_shapes=True)
        # tf.keras.utils.plot_model(self.model_C, 'Critic Network.png', show_shapes=True)
        # tf.keras.utils.plot_model(self.model_AC, 'Actor-Critic Network.png', show_shapes=True)

    def get_weights(self):
        return self.model_AC.get_weights()

    def forward_p(self, state):
        # forward propagation
        return self.model_AC(state)

    def receive_weights(self, pipe, num):
        stop_counter = 0
        while stop_counter < num:
            receive = pipe.recv()
            if receive == True:
                stop_counter += 1
            else:
                d_A_acc = receive[0]
                d_C_acc = receive[1]
                self.opti_A.apply_gradients(zip(d_A_acc, self.model_A.trainable_weights))
                self.opti_C.apply_gradients(zip(d_C_acc, self.model_C.trainable_weights))
                pipe.send(self.model_AC.get_weights())



class LocalAgent:
    def __init__(self, weights, index, seed):
        # args:
        # weights: weights of center agent
        # index: index of this local agent

        self.index = index

        # create game:
        self.env = gym.make('LunarLanderContinuous-v2')
        self.env.seed(seed=seed)

        # create Actor-Critic network:
        # 1. create Actor network:
        inputs = tf.keras.Input(shape=(9,))
        x = tf.keras.layers.Dense(UNIT_ACTOR_1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)  # 1st layer
        x = tf.keras.layers.Dense(UNIT_ACTOR_2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # 2nd layer
        outputs_mu = tf.keras.layers.Dense(2, activation='tanh')(x)  # 2 means (corresponding to 2 actions) in 3rd layer (linear)
        outputs_sigma2 = tf.keras.layers.Dense(2, activation='softplus', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # 1 variance in 3rd layer (with square)

        self.model_A = tf.keras.Model(inputs=inputs,
                                      outputs=[outputs_mu, outputs_sigma2],
                                      name='Actor Network')

        # 2. create Critic network:
        y = tf.keras.layers.Dense(UNIT_CRITIC_1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)  # 1st layer
        y = tf.keras.layers.Dense(UNIT_CRITIC_2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(y)  # 2nd layer
        outputs_c = tf.keras.layers.Dense(1)(y)  # 3rd layer

        self.model_C = tf.keras.Model(inputs=inputs,
                                      outputs=outputs_c,
                                      name='Critic Network')

        # 3. create Actor-Critic network
        self.model_AC = tf.keras.Model(inputs=inputs,
                                       outputs=[outputs_mu, outputs_sigma2, outputs_c],
                                       name='Actor-Critic Network')

        # set weights (same as center agent):
        self.model_AC.set_weights(weights=weights)

        # tensorboard:
        # 1. define our metrics:
        #self.reward_tb = tf.keras.metrics.Mean('total reward', dtype=tf.float32)
        self.loss_tb = tf.keras.metrics.Mean('total reward', dtype=tf.float32)
        # 2. setup writers
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_dir = 'logs/' + current_time + '/reward'
        self.tb_writer = tf.summary.create_file_writer(self.tb_dir)

        self.counter = 0
        self.destroy = False


    def forward_p(self, state):
        # forward propagation:
        return self.model_AC(state)

    def train(self, lock, pipe):
        # train the network
        print('agent:', self.index, 'training begins')

        # reset the game:
        observation = self.env.reset()

        counter_step = 0  # step counter
        reward_total = 0.0  # total reward in one episode
        i_ep = 1

        while i_ep <= EPISODE_MAX:
            # logs:
            log_d_A = []  # log gradient of Actor network
            log_d_C = []  # log gradient of Critic network
            log_V = []  # log Value function (output of Critic network)
            log_r = []  # log rewards

            for i_step in range(STEP_MAX):
                counter_step += 1

                with tf.GradientTape() as t_A:
                    t_A.watch(self.model_A.trainable_weights)
                    observation_9 = np.append(observation, counter_step / 500 - 1)
                    outputs_A = self.model_A(observation_9.reshape(1,9))  # obtain the outputs of Actor network
                    # split outputs of Actor network:
                    mu_1 = outputs_A[0][0][0]
                    mu_2 = outputs_A[0][0][1]
                    sigma2_1 = outputs_A[1][0][0]
                    sigma2_2 = outputs_A[1][0][1]
                    #print('action(pred):', [mu_1.numpy(), mu_2.numpy()], 'standard deviation:', sigma2_1.numpy() ** 0.5)
                    #print('standard deviation 1:', sigma2_1.numpy() ** 0.5, 'standard deviation 2:', sigma2_2.numpy()**0.5)
                    # sample actions:
                    action_1 = np.clip(np.random.normal(loc=mu_1.numpy(),
                                                scale=sigma2_1.numpy()**0.5), -1, 1)

                    action_2 = np.clip(np.random.normal(loc=mu_2.numpy(),
                                                scale=sigma2_2.numpy()**0.5), -1, 1)

                    #print('action(true):', [action_1, action_2])
                    # calculate pdf:
                    pdf = 1/(3.14*2*(sigma2_1*sigma2_2)**0.5)*tf.math.exp(-(action_1-mu_1)**2/(2*sigma2_1)-(action_2-mu_2)**2/(2*sigma2_2))
                    if pdf.numpy()==0:
                        template = 'action_1: {}, action_2: {}, sigma2_1: {}, exp: {}'
                        print(template.format(action_1, action_2, sigma2_1.numpy(), tf.math.exp(-((action_1-mu_1)**2+(action_2-mu_2)**2)/(2*sigma2_1)).numpy()))
                    # calculate log(pdf):
                    pdf_log = tf.math.log(pdf) + BETA * ( tf.math.log(2*3.14*(sigma2_1)) + tf.math.log(2*3.14*(sigma2_2)) )

                # calculate gradient:
                #print('pdf:', pdf.numpy(), 'expl:', BETA * ( tf.math.log(2*3.14*(sigma2_1)) + tf.math.log(2*3.14*(sigma2_2)) ).numpy(), 'pdf_log:', pdf_log.numpy())
                d_A = t_A.gradient(pdf_log, self.model_A.trainable_weights)
                log_d_A.append(d_A)  # log gradient of Actor network
                #print(d_A[0][0].numpy())

                with tf.GradientTape() as t_C:
                    outputs_C = self.model_C(observation_9.reshape(1,9))  # obtain the outputs of Critic network

                # calculate gradient:
                d_C = t_C.gradient(outputs_C, self.model_C.trainable_weights)
                log_d_C.append(d_C)  # log gradient of Critic network
                log_V.append((outputs_C.numpy()[0][0]))  # log Value function

                # act in game:
                observation, reward, done, _ = self.env.step([action_1, action_2])
                log_r.append(reward)  # log rewards
                reward_total += reward  # add reward to total reward
                self.env.render()  # render the game

                if i_step == STEP_MAX-1:
                    observation_9 = np.append(observation, counter_step / 500 - 1)
                    value_last = self.model_C(observation_9.reshape(1, 9)).numpy()[0][0]  # store the last Value

                # check done:
                if done:
                    # tensorboard:
                    #self.reward_tb(reward_total)
                    with self.tb_writer.as_default():
                        tf.summary.scalar('reward '+str(self.index), reward_total, step=i_ep)

                    observation = self.env.reset()  # reset the game
                    template = '=======================\nagent {}: done with {} step(s) in episode {} total reward {}'
                    print(template.format(self.index, counter_step, i_ep, reward_total))
                    if counter_step == 1000:
                        self.destroy = False
                    else:
                        self.destroy = True

                    counter_step = 0  # reset the step counter
                    reward_total = 0
                    i_ep += 1  # one episode done

                    break

            print('action(pred):', [mu_1.numpy(), mu_2.numpy()], 'standard deviation:', sigma2_1.numpy() ** 0.5)
            # calculate accumulate reward:
            if done :# and self.destroy:
                reward_acc = 0.0
            else:
                reward_acc = value_last

            for i in range(len(log_r)):
                reward_acc = reward_acc * GAMMA + log_r[-(i+1)]
                advantage = reward_acc - log_V[-(i+1)]  # advantage function
                # tensorboard:
                self.loss_tb(advantage**2)
                with self.tb_writer.as_default():
                    tf.summary.scalar('loss ' + str(self.index), advantage**2, step=self.counter)
                self.counter += 1
                if i==0:
                    d_A_acc =[0 for _ in range(len(d_A))]
                    d_C_acc =[0 for _ in range(len(d_C))]
                for j in range(len(d_A)):
                    d_A_acc[j] += - advantage * log_d_A[-(i+1)][j]
                for j in range(len(d_C)):
                    d_C_acc[j] += - 2 * advantage * log_d_C[-(i+1)][j]

            # upload gradients:
            lock.acquire()
            pipe.send([d_A_acc, d_C_acc])

            # download weights:
            weights_center = pipe.recv()
            self.model_AC.set_weights(weights_center)

            # indicate precessing done:
            if i_ep > EPISODE_MAX:
                pipe.send(True)

            lock.release()

            # apply gradients to Actor-Critic network:
            #self.opti_A.apply_gradients(zip(d_A_acc, self.model_A.trainable_weights))
            #self.opti_C.apply_gradients(zip(d_C_acc, self.model_C.trainable_weights))

def local_run(weights, index, seed, lock, pipe):
    agent_local = LocalAgent(weights=weights, index=index, seed=seed)
    agent_local.train(lock=lock, pipe=pipe)

if __name__ == '__main__':
    agent_center = CenterAgent()

    lock = mp.Lock()

    center_end, local_end = mp.Pipe()

    p1 = mp.Process(target=local_run, args=(agent_center.get_weights(), 1, 10, lock, local_end))
    p2 = mp.Process(target=local_run, args=(agent_center.get_weights(), 2, 23, lock, local_end))
    p3 = mp.Process(target=local_run, args=(agent_center.get_weights(), 3, 53, lock, local_end))
    p1.start()
    p2.start()
    p3.start()

    agent_center.receive_weights(pipe=center_end, num=3)

