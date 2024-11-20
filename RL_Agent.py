import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from gym import Env
from gym.spaces import Box


class GraphEnv(Env):
    def __init__(self, graph):
        super(GraphEnv, self).__init__()
        self.graph = graph
        self.num_nodes = len(graph.nodes)
        self.action_space = Box(low=-1, high=1, shape=(self.num_nodes, 2),
                                dtype=np.float32)  # Action: Adjust node positions
        self.observation_space = Box(low=-1, high=1, shape=(self.num_nodes, 2),
                                     dtype=np.float32)  # Obs: Current positions
        self.positions = np.random.rand(self.num_nodes, 2)  # Initialize random positions

    def step(self, action):
        self.positions += action
        reward = -self.compute_crossings()  # Reward = negative edge crossings
        done = True  # Single step per episode
        return self.positions, reward, done, {}

    def reset(self):
        self.positions = np.random.rand(self.num_nodes, 2)
        return self.positions

    def compute_crossings(self):
        # Placeholder: Simple counting for illustration, replace with actual geometry-based crossings check
        crossings = 0
        edges = list(self.graph.edges)
        for i, (u1, v1) in enumerate(edges):
            for j, (u2, v2) in enumerate(edges):
                if i < j and self.edge_intersect(u1, v1, u2, v2):
                    crossings += 1
        return crossings

    def edge_intersect(self, u1, v1, u2, v2):
        # Placeholder logic: Replace with geometric intersection check
        return np.random.choice([True, False])


# Wrap with DummyVecEnv for Stable-Baselines3 compatibility
env = DummyVecEnv([lambda: GraphEnv(G)])
