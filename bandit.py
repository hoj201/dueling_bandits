"""Model stores bandit code"""
import numpy as np
from abc import ABC
from scipy.stats import beta
from enum import Enum, auto
from itertools import product
from typing import List, Optional


class Action(Enum):
    pass


class Bandit(ABC):
    def act(self) -> Action:
        pass

    def update(self, action, reward):
        pass


class BetaDistribution(object):
    def __init__(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta
        self.history = []

    def update(self, reward: bool):
        self.history.append(reward)
        if len(self.history) > 10:
            self.history = self.history[1:]
        self.alpha = sum([1 for x in self.history if x]) + 0.5
        self.beta = sum([1 for x in self.history if not x]) + 0.5

    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    def sample(self, n_samples=None) -> np.float64:
        rv = beta(self.alpha, self.beta)
        if n_samples is None:
            return rv.rvs()
        return rv.rvs(n_samples)

    def __repr__(self):
        return f"BetaDistribution({self.alpha}, {self.beta})"


class BetaBernoulliBandit(Bandit):
    def __init__(self, actions: List[Action], contexts: Optional[List] = None):
        if contexts is None:
            contexts = [None]
        self.contexts = contexts
        self.actions = actions
        self.beta_distributions = dict()
        for ctx, a in product(self.contexts, self.actions):
            self.beta_distributions[(ctx, a)] = BetaDistribution()

    def act(self, context=None):
        """Sample from beta distributions"""
        out = None
        highest_expectation = 0.0
        for a in self.actions:
            theta = self.beta_distributions[(context, a)].sample()
            if theta > highest_expectation:
                highest_expectation = theta
                out = a
        return out

    def prob(self, context=None, n_samples=100):
        """Computes the probability of choosing a lever in a given context"""
        samples = {
            a: self.beta_distributions[(context, a)].sample(n_samples)
            for a in self.actions
        }
        actions = list(self.actions)
        n_0 = np.count_nonzero(samples[actions[0]] > samples[actions[1]])
        n_1 = n_samples - n_0
        return {actions[0]: n_0 / n_samples, actions[1]: n_1 / n_samples}

    def update(self, action: Action, reward: bool, context=None):
        self.beta_distributions[(context, action)].update(reward)


if __name__ == "__main__":
    class ChildAction(Action):
        PeeOnFloor = auto()
        PeeInToilet = auto()


    child = BetaBernoulliBandit(actions=[ChildAction.PeeOnFloor, ChildAction.PeeInToilet])
    for _ in range(10):
        action = child.act()
        reward = (action == ChildAction.PeeInToilet)
        print(action, reward)
        child.update(action, reward)
    for (context, action), dist in child.beta_distributions.items():
        print(f"{action.name}: {dist}")
    print(child.beta_distributions)
