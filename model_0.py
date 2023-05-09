"""In this model we just reward peeing in the toilet"""
from bandit import Action, BetaBernoulliBandit
from enum import auto


class ChildAction(Action):
    PeeInToilet = auto()
    PeeOnFloor = auto()


if __name__ == "__main__":
    child = BetaBernoulliBandit([ChildAction.PeeInToilet, ChildAction.PeeOnFloor])
    pee_in_toilet_prob = []
    for _ in range(100):
        pee_in_toilet_prob.append(
            child.prob(n_samples=200)[ChildAction.PeeInToilet]
        )
        action = child.act()
        reward = (action == ChildAction.PeeInToilet)
        child.update(action, reward)
    print(child.beta_distributions)
    import json
    with open ("model_outputs/model_0.json", "w") as f:
        json.dump(pee_in_toilet_prob, f)
