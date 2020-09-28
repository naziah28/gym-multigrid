import gym
import time
from gym.envs.registration import register
import argparse

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--env', default='soccer', type=str)

args = parser.parse_args()

def main():

    if args.env == 'putnear':
        register(
            id='multigrid-putnear-v0',
            entry_point='gym_multigrid.envs:PutNearEnv12x12N2',
        )
        env = gym.make('multigrid-putnear-v0')

    else:
        register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
        )
        env = gym.make('multigrid-collect-v0')

    _ = env.reset()

    nb_agents = len(env.agents)

    while True:
        env.render(mode='human')
        time.sleep(0.1)

        ac = [env.action_space.sample() for _ in range(nb_agents)]

        obs, rewards, done, info = env.step(ac)

        # print("\nactions:", ac)
        # print("Obs: {}\nRewards: {}\nInfo: {}".format(obs[0].shape, rewards, info))

        if done:
            break

if __name__ == "__main__":
    main()