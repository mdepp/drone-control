import gym
from gym_environment import GymEnvironment

def main():
    environment = GymEnvironment(gym.make('Cartpole-v1'))

if __name__ == '__main__':
    main()
