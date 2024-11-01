import gym
from gym import spaces
import numpy as np
import argparse
import sys

class JobEnv(gym.Env):
    """Job environment where a job can crash and must be observed to monitor its state."""
    metadata = {'render.modes': ['human']}

    def __init__(self, T, P, C, max_steps=100, initial_reward=None, seed=None):
        """
        Initialize the environment.
        
        Parameters:
        - T: int, total time steps required to complete the job if no crashes occur.
        - P: float, probability of the job completing
        - c: float, cost parameter for observing the job.
        """
        super(JobEnv, self).__init__()

        self.T = T                       # Time required to complete the job
        self.p = 1 - (P**(1/T))          # Probability of crash per time step
        self.observe_cost = C                       # Observation cost parameter
        self.max_steps = max_steps
        self.initial_reward = initial_reward if initial_reward is not None else (-1 / max_steps)
        self.seed = seed

        # Define action space: 0 for "wait" and 1 for "observe"
        self.action_space = spaces.Discrete(2)

        # Observation space: job state (remaining time if job is crashed, else 0)
        self.observation_space = spaces.Discrete(T + 1)

    def reset(self):
        """Reset the environment to the initial state."""
        if self.seed is not None:
            np.random.seed(self.seed)

        self.time_left = self.T
        self.crashed = False
        self.steps_taken = 0
        return self.time_left  # Initial observation

    def step(self, action):
        """
        Take an action in the environment.
        
        Parameters:
        - action: int, 0 for "wait" and 1 for "observe"
        
        Returns:
        - observation: int, time remaining (or T if crashed)
        - reward: float, reward obtained after taking the action
        - done: bool, whether the episode has ended
        - info: dict, additional information
        """
        self.steps_taken += 1
        observation = self.T
        reward = self.initial_reward
        done = False

        # Check if job crashes this step
        if np.random.random() < self.p:
            self.crashed = True

        if (not self.crashed) and (self.time_left > 0):
            self.time_left -= 1

        # If action is "observe"
        if action == 1:
            reward -= self.observe_cost

            # Check if job is complete
            if not self.crashed and (self.time_left == 0):
                reward += 1  # Reward for successfully completing the job
                done = True

            # Observe and check if crashed, reset if necessary
            if self.crashed:
                self.time_left = self.T  # Restart job on crash
                self.crashed = False

            observation = self.time_left

        return observation, reward, done, {}


    def render(self, mode='human'):
        """Render the environment."""
        status = "Crashed" if self.crashed else "Running"
        print(f'Step: {self.steps_taken}, Time Left: {self.time_left}, Status: {status}')

    def close(self):
        """Close the environment."""
        pass

def validate_args(args):
    T, P, C = args.T, args.P, args.C

    if not (isinstance(T, int) and T > 0):
        print("Error: T must be a positive integer.")
        sys.exit(1)
    if not (0 <= P <= 1):
        print("Error: P must be a float in the range 0 <= P <= 1.")
        sys.exit(1)
    if not (0 <= C < 1):
        print("Error: C must be a float in the range 0 <= C < 1.")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process parameters T, P, and C. Also can take in num_iters")
    
    parser.add_argument("-T", type=int, default=1, help="A positive integer for T")
    parser.add_argument("-P", type=float, default=1.0, help="A float in the range 0 <= P <= 1")
    parser.add_argument("-C", type=float, default=0.1, help="A float in the range 0 <= C < 1")
    parser.add_argument("--num_iters", type=int, default=100, help="The number of iterations to run")

    args = parser.parse_args()

    # Validate the arguments
    validate_args(args)

    # Use T, p, c in the rest of the program
    print(f"T: {args.T}, p: {args.P}, c: {args.C}")
    env = JobEnv(T=args.T, P=args.P, C=args.C)

    total_rewards = []
    total_steps_taken = []
    
    print(f"Running {args.num_iters} iterations\n")
    for _ in range(args.num_iters):
        env.reset()
        done = False
        total_reward = 0
        while (not done):
            _, reward, done, _ = env.step(1)
            total_reward += reward
        total_rewards.append(total_reward)
        total_steps_taken.append(env.steps_taken)

    print(f"Reward: {np.mean(total_rewards):.3f} +/- {np.std(total_rewards):.3f} with range {np.min(total_rewards)} - {np.max(total_rewards)}")
    print(f"Steps Taken: {np.mean(total_steps_taken):.3f} +/- {np.std(total_steps_taken):.3f} with range {np.min(total_steps_taken)} - {np.max(total_steps_taken)}")


