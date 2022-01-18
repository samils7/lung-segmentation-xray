from cProfile import label
import os
import json
import matplotlib.pyplot as plt


def plot(data:dict, title:str, name:str):
    plt.rcParams['figure.figsize'] = [15, 4]
    plt.figure()
    modes = ["train", "val"]
    scores = ["loss", "iou", "acc"]
    length = len(scores)

    for i, score in enumerate(scores):
        plt.subplot(1, length, i + 1)
        for mode in modes:
            plt.plot(data[mode]["epoch"], data[mode][score], label=mode)
        plt.title(score)
        plt.legend()
        plt.grid()
    
    plt.suptitle(title)
    plt.savefig(f"{name}.png")
    plt.savefig(f"{name}.pdf")




if __name__ == "__main__":
    base = "results"
    for ciriterion in os.listdir(f"{base}"):
        for augmentation in os.listdir(f"{base}/{ciriterion}"):
            path = f"{base}/{ciriterion}/{augmentation}/results.json"
            print(path)
            data = json.load(open(path, "r"))
            plot(data, f"{ciriterion} {augmentation}", f"{base}/{ciriterion}/{augmentation}/logs")
        
