from environment import JobEnv

import numpy as np

class ObserveGapOptimizer:
    def __init__(self, env):
        """
        Initialize the POMDP solver.
        
        Parameters:
        - env: JobEnv, the POMDP environment
        - observe_gap: int, the number of time steps to wait before observing
        """
        self.env = env
        # self.observe_gap = observe_gap

    def run_episode(self, observe_gap):
        """Run a single episode using the observe_gap policy."""
        self.env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            # Decide action based on observe_gap
            if steps % observe_gap == 0:
                action = 1  # Observe
            else:
                action = 0  # Wait

            # Take action in the environment
            reward, done, _ = self.env.step(action)
            total_reward += reward
            steps += 1

        return total_reward, steps

    def evaluate_policy(self, num_episodes=100):
        """Evaluate the observe_gap policy over multiple episodes."""
        total_rewards = []
        # total_steps = []

        for _ in range(num_episodes):
            reward, _ = self.run_episode()
            total_rewards.append(reward)
            # total_steps.append(steps)

        avg_reward = np.mean(total_rewards)
        # reward_std = np.std(total_rewards)
        # avg_steps = np.mean(total_steps)
        # steps_std = np.std(total_steps)
        return avg_reward

        # print(f"Average reward: {avg_reward:.3f} +/- {reward_std:.3f}")
        # print(f"Average steps taken: {avg_steps:.3f} +/- {steps_std:.3f}")

    def solve(self):
        for observe_gap in range()
    

# Example usage
if __name__ == '__main__':
    # Example environment parameters

    T = 10

    T, P, C = 10, 0.9, 0.05
    env = JobEnv(T=T, P=P, C=C)

    # Initialize the solver with an observe_gap parameter
    solver = ObserveGapOptimizer(env=env)
    observe_gap = solver.solve()
