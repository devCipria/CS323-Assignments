# CSCI 323
# Winter 2022
# Assignment 1 (Search Algorithms)
# David Cipriati

import math
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def random_list(max_int, size, do_sort=False):
    # Gives you a random array of size 'size'
    arr = [random.randint(1, max_int) for i in range(size)]
    if do_sort:
        arr.sort()
    return arr


# Linear Search - https://www.geeksforgeeks.org/linear-search/
def linear_search(arr, key):
    for i in range(0, len(arr)):
        if arr[i] == key:
            return i
    return -1


# Iterative Binary Search - from the book Grokking Algorithms by Aditya Bhargava
def binary_search(arr, key):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        guess = arr[mid]

        if guess == key:
            return mid
        if guess > key:
            high = mid - 1
        else:
            low = mid + 1
    return -1


def interpolation_search(arr, key):
    return interpolation_search_recursive(arr, 0, len(arr) - 1, key)


# https://www.geeksforgeeks.org/interpolation-search/
def interpolation_search_recursive(arr, lo, hi, x):
    if arr[lo] == arr[hi]:
        if x == arr[lo]:
            return lo;
        else:
            return -1

    # Since array is sorted, an element present in array must be in range defined by corner
    if lo <= hi and arr[lo] <= x <= arr[hi]:

        # Probing the position with keeping uniform distribution in mind.
        pos = int(lo + ((hi - lo) / (arr[hi] - arr[lo]) * (x - arr[lo])))

        # Condition of target found
        if arr[pos] == x:
            return pos

        # If x is larger, x is in right subarray
        if arr[pos] < x:
            return interpolation_search_recursive(arr, pos + 1, hi, x)

        # If x is smaller, x is in left subarray
        if arr[pos] > x:
            return interpolation_search_recursive(arr, lo, pos - 1, x)
    return -1


# Jump Search - https://www.geeksforgeeks.org/jump-search/
def jump_search(arr, key):
    # Finding block size to be jumped
    step = math.sqrt(len(arr))

    # Finding the block where element is present (if it is present)
    prev = 0
    while arr[int(min(step, len(arr)) - 1)] < key:
        prev = step
        step += math.sqrt(len(arr))
        if prev >= len(arr):
            return -1

    # Doing a linear search for key in block beginning with prev.
    while arr[int(prev)] < key:
        prev += 1

        # If we reached next block or end of array, element is not present.
        if prev == min(step, len(arr)):
            return -1

    # If element is found
    if arr[int(prev)] == key:
        return int(prev)

    return -1


def python_search(arr, key):
    return arr.index(key)


def plot_times_line_graph(dict_searches):
    for search in dict_searches:
        x = dict_searches[search].keys()
        y = dict_searches[search].values()
        plt.plot(x, y, label=search)
    plt.legend()
    plt.title("Run Time of Search Algorithms")
    plt.xlabel("Number of Elements")
    plt.ylabel("Time for 100 Trials (ms)")
    plt.savefig("search_graph.png")
    plt.show()


def plot_times_bar_graph(dict_searches, sizes, searches):
    search_num = 0
    plt.xticks([j for j in range(len(sizes))], [str(size) for size in sizes])
    for search in searches:
        search_num += 1
        d = dict_searches[search.__name__]
        x_axis = [j + 0.05 * search_num for j in range(len(sizes))]
        y_axis = [d[i] for i in sizes]
        plt.bar(x_axis, y_axis, width=.05, alpha=.25, label=search.__name__)
    plt.legend()
    plt.title("Run Time of Search Algorithms")
    plt.xlabel("Number of Elements")
    plt.ylabel("Time for 100 Trials (ms)")
    plt.savefig("search_graph_bar.png")
    plt.show()


def main():
    max_int = 1000000
    trials = 10
    dict_searches = {}
    searches = [linear_search, binary_search, interpolation_search, jump_search, python_search]
    for search in searches:
        dict_searches[search.__name__] = {}
    sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    for size in sizes:
        for search in searches:
            dict_searches[search.__name__][size] = 0
        for trial in range(1, trials):
            arr = random_list(max_int, size, True)
            idx = random.randint(1, size) - 1
            key = arr[idx]
            for search in searches:
                start_time = time.time()
                idx2 = search(arr, key)
                end_time = time.time()
                net_time = end_time - start_time
                dict_searches[search.__name__][size] += 1000 * net_time
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    df = pd.DataFrame.from_dict(dict_searches).T
    print(df)
    # plot_times_line_graph(dict_searches)
    plot_times_bar_graph(dict_searches, sizes, searches)


if __name__ == '__main__':
    main()

