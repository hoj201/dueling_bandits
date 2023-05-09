"""In this model the parent is a bandit as well as the child"""
from bandit import Action, BetaBernoulliBandit
from enum import auto


class ChildAction(Action):
    PeeOnFloor = auto()
    PeeInToilet = auto()


class ParentAction(Action):
    GiveCandy = auto()
    Frown = auto()


if __name__ == "__main__":
    child = BetaBernoulliBandit(
        actions=[ChildAction.PeeInToilet, ChildAction.PeeOnFloor]
    )
    parent = BetaBernoulliBandit(
        actions=[ParentAction.GiveCandy, ParentAction.Frown],
        contexts=[ChildAction.PeeOnFloor, ChildAction.PeeInToilet]
    )

    parent_action = None
    pee_in_toilet_prob = []
    for _ in range(100):
        pee_in_toilet_prob.append(
            child.prob(n_samples=200)[ChildAction.PeeInToilet]
        )
        child_action = child.act()
        if parent_action is not None:
            parent_reward = (child_action == ChildAction.PeeInToilet)
            parent.update(parent_action, parent_reward, context=child_action)
        parent_action = parent.act(context=child_action)
        child_reward = (parent_action == ParentAction.GiveCandy)
        child.update(child_action, child_reward)
    import json
    with open("model_outputs/model_2.json", "w") as f:
        json.dump(pee_in_toilet_prob, f)

