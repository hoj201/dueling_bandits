"""In this model the parent has an internal model of the child who may or may not be a scrambled"""
from bandit import BetaBernoulliBandit, Action, auto
import numpy as np


class ChildAction(Action):
    PeeOnFloor = auto()
    PeeInToilet = auto()


class ParentAction(Action):
    GiveCandy = auto()
    Frown = auto()


class Parent(object):
    def __init__(self):
        self.child = BetaBernoulliBandit(
            actions=[ChildAction.PeeInToilet, ChildAction.PeeOnFloor]
        )
        self.scrambled_child = BetaBernoulliBandit(
            actions=[ChildAction.PeeInToilet, ChildAction.PeeOnFloor]
        )

    def child_is_scrambled(self, observed_action) -> bool:
        """randomly return True or false by .... Pr( child_is_scrambled | something)"""
        scrambled = (np.random.rand() > 0.5)
        if scrambled and self.scrambled_child.act() == observed_action:
            return True
        if not scrambled and self.child.act() == observed_action:
            return False
        return self.child_is_scrambled(observed_action)

    def act(self, context: ChildAction, reward=None) -> ParentAction:
        if reward is None:
            reward = (context == ChildAction.PeeInToilet)
            scrambled = self.child_is_scrambled(context)
            print(f"child scrambled predicted {scrambled}")
            if scrambled:
                reward = not reward
        self.child.update(action=context, reward=reward)
        self.scrambled_child.update(action=context, reward=not reward)
        response = ParentAction.GiveCandy if reward else ParentAction.Frown
        return response


if __name__ == "__main__":
    child = BetaBernoulliBandit(
        actions=[ChildAction.PeeInToilet, ChildAction.PeeOnFloor]
    )
    parent = Parent()
    reward = None
    parent_action = None
    pee_in_toilet_probabilities = []
    for it in range(100):
        pee_in_toilet_probabilities.append(
            child.prob(n_samples=200)[ChildAction.PeeInToilet]
        )
        child_action = child.act()
        parent_action = parent.act(context=child_action, reward=reward)
        child_reward = (parent_action == ParentAction.GiveCandy)
        child.update(action=child_action, reward=not child_reward)

    import json
    with open("model_outputs/model_3.json", "w") as f:
        json.dump(pee_in_toilet_probabilities, f)
