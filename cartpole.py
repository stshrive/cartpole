import torch
import torch.nn.functional as F
import numpy as np
import random
import gym

from torch import nn
from collections import deque
from collections import abc

import ttypes

class QNetwork(nn.Module):
    def __init__(self, _in: int, _out: int, _hidden: int):
        super(QNetwork, self).__init__()
        self.inputs  = _in
        self.outputs = _out
        self.W1 = nn.Linear(_in, _hidden)
        self.W2 = nn.Linear(_hidden, _out)

    def forward(self, _in):
        _in = ttypes.from_numpy(_in)
        out = self.W1(_in)
        out = F.relu(out)
        out = self.W2(out)
        out = F.softmax(out, dim=-1)
        return out.cpu().detach().numpy()

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
        self.storage.append(tuple(args))

    def sample(self, size: int):
        return random.sample(self.storage, size)

    def __len__(self):
        return len(self.storage)

class Agent:
    def __init__(self, model: nn.Module, memory: Memory,
            discount_rate: float,
            random_threshold: float,
            threshold_decay: float,
            minimum_threshold: float,
            *args,
            **kwargs
        ):
        self.model  = model
        self.memory = memory

        self.discount_rate     = discount_rate
        self.random_threshold  = random_threshold
        self.threshold_decay   = threshold_decay
        self.minimum_threshold = minimum_threshold

    def choose_action(self, state):
        if np.random.rand() <= self.random_threshold:
            return ttypes.LongTensor([[random.randrange(self.model.outputs)]])
        else:
            action = self.model(state)
            return self.model(state).max(0)[1].view(1, 1)

    def remember(self, state, action, reward, next_state, terminal):
        self.memory.commit(state, action, reward, next_state, terminal)

    def replay(self, batch_size: int):
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)

        while batch:
            state, action, reward, next_state, terminal = batch.pop(0)
            target = reward
            if not terminal:
                target = reward + self.discount_rate \
                        * self.model(next_state)
            future_discount = self.model(state)

            if terminal:
                break

            self.model.optimize(future_discount, target, F.smooth_l1_loss)
        if self.random_threshold > self.minimum_threshold:
            self.random_threshold *= self.threshold_decay


# This is where the agent plays in an environement to learn
class GymRunner:
    def __init__(self, environment: gym.Env, agent: Agent):
        self.environment = environment
        self.agent = agent

    def execute(self, episode_length: int):
        state, reward = self.environment.reset(), 0
        agent = self.agent

        ending_frame = 0

        for episode_frame in range(episode_length):
            action = self.agent.choose_action(state)
            next_state, reward, terminal, _ = self.environment.step(action)

            self.agent.remember(state, action, reward, next_state, terminal)
            state = next_state

            if terminal:
                ending_frame = episode_frame
                break

        agent.replay(32)
        return ending_frame

def main(environment: str, epochs: int, hyper_parameters: HyperParameters):
    environment = gym.make(environment)

    model = QNetwork(
            environment.observation_space.shape[0],
            environment.action_space.n,
            hyper_parameters.hidden)

    agent = Agent(
            model,
            Memory(hyper_parameters.capacity),
            **hyper_parameters)

    sandbox = GymRunner(environment, agent)

    for epoch in range(epochs):
        t = sandbox.execute(hyper_parameters.episode_length)
        print(f'epoch: {epoch}/{epochs}, SCORE: {t}')


if __name__ == "__main__":
    import sys
    import argparse

    argp = argparse.ArgumentParser(sys.argv[0])
    argp.add_argument('--environment', '-g', type=str, default="CartPole-v0")
    argp.add_argument('--discount_rate', '-G', type=float, default=0.95)
    argp.add_argument('--random_threshold', '-E', metavar='EXPLORATION_RATE', type=float, default=1.0)
    argp.add_argument('--threshold_decay', '-D', type=float, default=0.995)
    argp.add_argument('--minimum_threshold', '-M', type=float, default=0.01)
    argp.add_argument('--capacity', '-c', type=int, default=10000)
    argp.add_argument('--episode_length', '-t', type=int, default=300)
    argp.add_argument('--epochs', '-e', type=int, default=10000)
    argp.add_argument('--hidden', '-H', type=int, default=128)

    args = argp.parse_args()
    params = HyperParameters(**args.__dict__)

    main(args.environment, args.epochs, params)
