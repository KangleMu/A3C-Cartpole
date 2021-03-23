import gym
import numpy as np

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    print(env.action_space)

    observation = env.reset()

    for i in range(1000):
        env.render()

        action = np.random.normal(loc=0, scale=1, size=2)
        observation, reward, done, _ = env.step(action)

        tamplate = 'observation: {}, reward: {}, done: {}'
        print(tamplate.format(observation, reward, done))

        if done:
            env.reset()

