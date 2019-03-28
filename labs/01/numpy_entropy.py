#!/usr/bin/env python3
# 96662533-4252-11e9-b0fd-00505601122b
import numpy as np

if __name__ == "__main__":
    data_dist = {}
    # Load data distribution, each data point on a line
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            if data_dist.get(line) == None:
                data_dist[line] = 0
            
            data_dist[line] = data_dist[line] + 1

    data = np.array(list(data_dist.values()))
    data = data / np.sum(data)

    # Load model distribution, each line `word \t probability`.
    model_probs = {}
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            [word, prob] = line.split("\t")
            model_probs[word] = float(prob)

    model = np.array([model_probs.get(k) or 0 for k in data_dist])

    entropy = -np.sum(data * np.log(data))
    print("{:.2f}".format(entropy))
    cross_entropy = -np.sum(data * np.log(model))
    print("{:.2f}".format(cross_entropy))
    d_kl = cross_entropy - entropy
    print("{:.2f}".format(d_kl))
