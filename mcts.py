from environment import JobEnv

import numpy as np
import math
import random

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # The state (observation, remaining time, crashed status)
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == 2  # "observe" and "wait" actions

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / (child.visits + 1)) + c_param * math.sqrt((2 * math.log(self.visits + 1) / (child.visits + 1)))
            for child in self.children.values()
        ]
        return list(self.children.values())[np.argmax(choices_weights)]

    def expand(self, action, new_state):
        if action not in self.children:
            self.children[action] = MCTSNode(new_state, parent=self)
        return self.children[action]

    def update(self, reward):
        self.visits += 1
        self.value += reward

class MCTS:
    def __init__(self, env, num_simulations=100, max_depth=20, gamma=0.99):
        self.env = env
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.gamma = gamma

    def run(self, root_state):
        root_node = MCTSNode(root_state)

        for _ in range(self.num_simulations):
            node = root_node
            state = root_state
            depth = 0
            done = False

            # Selection and Expansion
            while node.is_fully_expanded() and depth < self.max_depth:
                node = node.best_child()
                action = list(node.children.keys())[np.argmax([child.visits for child in node.children.values()])]
                state, reward, done, _ = self.env.step(action)
                depth += 1

            # Expansion
            if not node.is_fully_expanded() and not done:
                action = 0 if len(node.children) == 0 else 1
                observation, reward, done, _ = self.env.step(action)
                state = (observation, self.env.time_left, self.env.crashed)
                node = node.expand(action, state)

            # Simulation
            reward = self.rollout(state, depth)
            
            # Backpropagation
            self.backpropagate(node, reward)

        # Select the best action based on visit count
        best_action = max(root_node.children.items(), key=lambda child: child[1].visits)[0]
        return best_action

    def rollout(self, state, depth):
        total_reward = 0
        discount = 1

        while depth < self.max_depth:
            action = random.choice([0, 1])  # Random action during simulation
            observation, reward, done, _ = self.env.step(action)
            total_reward += discount * reward
            discount *= self.gamma
            depth += 1

            if done:
                break

        return total_reward

    def backpropagate(self, node, reward):
        while node is not None:
            node.update(reward)
            reward *= self.gamma  # Apply discount
            node = node.parent

# Example usage
if __name__ == '__main__':
    T, p, c = 10, 0.1, 0.05  # Example environment parameters
    env = JobEnv(T=T, p=p, c=c)
    
    # Initial state: observation (time left or T), job time left, and crashed status
    initial_state = (env.reset(), env.time_left, env.crashed)
    mcts_solver = MCTS(env, num_simulations=100, max_depth=20, gamma=0.99)
    
    total_rewards = []
    for _ in range(100):
        env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = mcts_solver.run(initial_state)
            _, reward, done, _ = env.step(action)
            total_reward += reward
            initial_state = (env.time_left, env.time_left, env.crashed)
        
        total_rewards.append(total_reward)
    
    print(f"Average reward: {np.mean(total_rewards):.3f} +/- {np.std(total_rewards):.3f}")
    print(f"Reward range: {np.min(total_rewards)} to {np.max(total_rewards)}")
