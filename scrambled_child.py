"""In this model the child scrambles the rewards"""
from bandit import Action, BetaBernoulliBandit
from enum import auto


class ChildAction(Action):
    PeeInToilet = auto()
    PeeOnFloor = auto()


if __name__ == "__main__":
    child = BetaBernoulliBandit([ChildAction.PeeInToilet, ChildAction.PeeOnFloor])
    pee_in_toilet_probability = []
    for _ in range(100):
        pee_in_toilet_probability.append(
            child.prob(n_samples=200)[ChildAction.PeeInToilet]
        )
        action = child.act()
        reward = (action == ChildAction.PeeInToilet)
        child.update(action, not reward)
    import json
    with open("model_outputs/model_1.json", "w") as f:
        json.dump(pee_in_toilet_probability, f)
