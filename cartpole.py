import torch
import torch.nn.functional as F
import numpy as np
import random
import gym

from torch import nn
from collections import deque
from collections import abc

class QNetwork(nn.Module):
    def __init__(self, _in: int, _out: int, _hidden: int):
        super(QNetwork, self).__init__()
        self.inputs  = _in
        self.outputs = _out
        self.W1 = nn.Linear(_in, _hidden)
        self.W2 = nn.Linear(_hidden, _out)

    def forward(self, _in):
        _in = torch.from_numpy(_in).float()
        out = self.W1(_in)
        out = F.relu(out)
        out = self.W2(out)
        out = F.softmax(out)
        return out.cpu().numpy()

    def optimize(self, state, target):
        pass

class HyperParameters(abc.Mapping):
    def __init__(self, **kwargs):
        self._properties = { k:kwargs[k] for k in kwargs }

    def __getattr__(self, attr):
        try:
            return self._properties[attr]
        except:
            raise AttributeError(
                f"'HyperParameters' object has no attribute '{attr}'"
            )

    def __getitem__(self, item):
        return self._properties[item]

    def __len__(self):
        return len(self._properties)

    def __iter__(self):
        return iter(self._properties)

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.storage  = []

    def commit(self, *args):
        if len(self.storage) >= self.capacity:
            self.storage.pop(0)
        self.storage.append(tuple(*args))

    def sample(self, size: int):
        return random.sample(self.storage, size)

class Agent:
    def __init__(self, model: nn.Module, memory: Memory,
            discount_rate: float,
            random_threshold: float,
            threshold_decay: float,
            minimum_threshold: float,
        ):
        self.model  = model
        self.memory = memory

        self.discount_rate     = discount_rate
        self.random_threshold  = random_threshold
        self.threshold_decay   = threshold_decay
        self.minimum_threshold = minimum_threshold

    def choose_action(self, state):
        if np.random.rand() <= self.random_threshold:
            return random.randrange(self.model.outputs)
        else:
            return np.argmax(self.model(state))

    def replay(self, batch_size: int):
        batch = self.memory.sample(batch_size)

        while batch:
            state, action, reward, next_state, terminal = batch.pop(0)
            target = reward
            if not terminal:
                target = reward + self.discount_rate \
                        * np.amax(self.model(next_state))
            future_discount = self.model(state)
            future_discount[0][action] = target
            self.model.optimize(state, future_discount)


# This is where the agent plays in an environement to learn
class GymRunner:
    def __init__(self, environment: gym.Env, agent: Agent):
        self.environment = environment
        self.agent = agent

    def execute(self):
        state, reward = self.environment.reset(), 0
