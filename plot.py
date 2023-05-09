from matplotlib import pyplot as plt
import json

if __name__ == "__main__":
    model = "model_3"
    with open(f"model_outputs/{model}.json", "r") as f:
        prob = json.load(f)

    plt.figure(figsize=(7, 3))
    plt.plot(range(len(prob)), prob, 'k')
    plt.grid(True)
    plt.xlabel("time")
    plt.ylabel("probability of desired action")
    plt.tight_layout()
    plt.savefig(f"plots/{model}.png")