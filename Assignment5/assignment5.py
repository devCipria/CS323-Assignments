# CSCI 323
# Winter 2022
# Assignment 5 (Bin Packing Problem)
# David Cipriati


import math
import random
import sys
import time
from datetime import datetime
from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join

capacity = 100


# https://www.geeksforgeeks.org/bin-packing-problem-minimize-number-of-used-bins/
def next_fit(weights):
    num_bins = 0
    rem_cap = capacity
    for _ in range(len(weights)):
        if rem_cap >= weights[_]:
            rem_cap = rem_cap - weights[_]
        else:
            num_bins += 1
            rem_cap = capacity - weights[_]
    return num_bins


# https://www.geeksforgeeks.org/bin-packing-problem-minimize-number-of-used-bins/
def first_fit(weights):
    n = len(weights)
    num_bins = 0
    bin_rem = [0] * n

    for i in range(n):
        j = 0
        while j < num_bins:
            if bin_rem[j] >= weights[i]:
                bin_rem[j] = bin_rem[j] - weights[i]
                break
            j += 1

        if j == num_bins:
            bin_rem[num_bins] = capacity - weights[i]
            num_bins = num_bins + 1

    return num_bins


# https://www.geeksforgeeks.org/bin-packing-problem-minimize-number-of-used-bins/
def best_fit(weights):
    n = len(weights)
    num_bins = 0
    bin_rem = [0] * n

    for i in range(n):
        j = 0

        minimum = capacity + 1
        best_bin_index = 0

        for j in range(num_bins):
            if (bin_rem[j] >= weights[i] and bin_rem[j] -
                    weights[i] < minimum):
                best_bin_index = j
                minimum = bin_rem[j] - weights[i]

        if minimum == capacity + 1:
            bin_rem[num_bins] = capacity - weights[i]
            num_bins += 1
        else:
            bin_rem[best_bin_index] -= weights[i]

    return num_bins


# https://www.geeksforgeeks.org/bin-packing-problem-minimize-number-of-used-bins/
def offline_ffd(weights):
    weights.sort(reverse=True)
    return first_fit(weights)


# https://www.geeksforgeeks.org/bin-packing-problem-minimize-number-of-used-bins/
def offline_bfd(weights):
    weights.sort(reverse=True)
    return best_fit(weights)


def optimal(weights):
    return math.ceil(reduce(lambda a, b: a + b, weights) / capacity)


def create_table(dict_weights):
    df = pd.DataFrame.from_dict(dict_weights).T
    print(df)
    print("\n")


# suggested by Edwin Ramos
def get_weights(file_path):
    weights = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            weights.append(int(line))
    return weights


def plot_bar_graph(dict_weights, algos, file_name, i):
    num_algo = 0
    plt.xticks([j for j in range(len(algos))], [algo.__name__ for algo in algos])
    for algo in algos:
        num_algo += 1
        d = dict_weights[file_name]
        x_axis = [num_algo-1]
        y_axis = [d[algo.__name__]]
        plt.bar(x_axis, y_axis, width=.75, alpha=.75, label=algo.__name__)
    # plt.legend()
    plt.title("BBP Approximation Algorithms Results: " + file_name)
    plt.xlabel("Algorithm")
    plt.ylabel("Number of Bins")
    plt.savefig("approx_algos_bar_graph" + str(i) + ".png")
    plt.show()


def main():
    mypath = "C:\\Users\\trade\\PycharmProjects\\323\\Assignment5\\BPPData\\"
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    trials = {}
    times = "Time for 40 Trials(ms)"
    trials[times] = {}
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    algos = [optimal, next_fit, first_fit, best_fit, offline_ffd, offline_bfd]

    for algo in algos:
        if algo.__name__ != "optimal":
            trials[times][algo.__name__] = 0

    for i, file in enumerate(files):
        dict_weights = {}
        i += 1
        weights = get_weights(mypath+file)
        print("Trial:", i, "\tItems:", len(weights))
        dict_weights[file] = {}

        for algo in algos:
            if algo.__name__ != "optimal":
                start_time = time.time()
                dict_weights[file][algo.__name__] = algo(weights)
                end_time = time.time()
                net_time = end_time - start_time
                trials[times][algo.__name__] += (net_time * 1000)
            else:
                dict_weights[file][algo.__name__] = algo(weights)

        if i % 10 == 0:
            plot_bar_graph(dict_weights, algos, file, i)

        create_table(dict_weights)

    create_table(trials)


if __name__ == '__main__':
    main()
