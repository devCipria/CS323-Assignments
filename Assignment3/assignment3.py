# CSCI 323
# Winter 2022
# Assignment 3 (String Search Algorithms)
# David Cipriati

import math
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string


# https://www.educative.io/edpresso/how-to-generate-a-random-string-in-python
def random_text(length):
    return ''.join(random.choice(string.ascii_uppercase) for i in range(0, length))


def random_pattern(pattern_length, text):
    idx = random.randint(0, len(text) - pattern_length)
    return text[idx:idx + pattern_length]


def native_search(pattern, text):
    idx = text.find(pattern)
    return idx


# https://www.geeksforgeeks.org/naive-algorithm-for-pattern-searching/
def brute_force(pattern, text):
    pattern_length = len(pattern)
    text_length = len(text)

    # A loop to slide pattern[] one by one */
    for i in range(text_length - pattern_length + 1):
        j = 0
        # For current index i, check for pattern match */
        while j < pattern_length:
            if text[i + j] != pattern[j]:
                break
            j += 1

        if j == pattern_length:
            return i


# https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/
def knuth_morris_pratt(pattern, text):
    pattern_length = len(pattern)
    text_length = len(text)

    # create lps[] that will hold the longest prefix, suffix values for pattern
    lps = [0] * pattern_length
    j = 0  # index for pattern[]

    # Preprocess the pattern (calculate lps[] array)
    compute_lps_array(pattern, pattern_length, lps)

    i = 0  # index for text[]
    while i < text_length:
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == pattern_length:
            # print("Pattern found at index " + str(i - j))
            return i - j
            j = lps[j - 1]

        # mismatch after j matches
        elif i < text_length and pattern[j] != text[i]:
            # Do not match lps[0..lps[j-1]] characters, they will match anyway
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1


# part of the Knuth-Morris Pratt function
def compute_lps_array(pat, M, lps):
    previous_length = 0  # length of the previous longest prefix suffix

    lps[0]  # lps[0] is always 0
    i = 1

    # the loop calculates lps[i] for i = 1 to M-1
    while i < M:
        if pat[i] == pat[previous_length]:
            previous_length += 1
            lps[i] = previous_length
            i += 1
        else:
            # This is tricky. Consider the example. AAACAAAA and i = 7. The idea is similar to search step.
            if previous_length != 0:
                previous_length = lps[previous_length - 1]

            # Also, note that we do not increment i here
            else:
                lps[i] = 0
                i += 1


# https://www.geeksforgeeks.org/rabin-karp-algorithm-for-pattern-searching/
def rabin_karp(pattern, text):
    # d is the number of characters in the input alphabet
    d = 256
    q = 101  # q is a prime number
    pattern_length = len(pattern)
    text_length = len(text)
    i = 0
    j = 0
    p = 0  # hash value for pattern
    t = 0  # hash value for text
    h = 1

    # The value of h would be "pow(d, M-1)%q"
    for i in range(pattern_length - 1):
        h = (h * d) % q

    # Calculate the hash value of pattern and first window of text
    for i in range(pattern_length):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q

    # Slide the pattern over text one by one
    for i in range(text_length - pattern_length + 1):
        # Check the hash values of current window of text and pattern if the hash values match then only check
        # for characters on by one
        if p == t:
            # Check for characters one by one
            for j in range(pattern_length):
                if text[i + j] != pattern[j]:
                    break
                else:
                    j += 1

            # if p == t and pattern[0...M-1] = text[i, i+1, ...i+M-1]
            if j == pattern_length:
                return i

        # Calculate hash value for next window of text: Remove leading digit, add trailing digit
        if i < text_length - pattern_length:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + pattern_length])) % q

            # We might get negative values of t, converting it to positive
            if t < 0:
                t = t + q


def plot_times_bar_graph(dict_searches, sizes, searches):
    search_num = 0
    plt.xticks([j for j in range(len(sizes))], [str(size) for size in sizes])
    for sort in searches:
        search_num += 1
        d = dict_searches[sort.__name__]
        x_axis = [j + 0.05 * search_num for j in range(len(sizes))]
        y_axis = [d[i] for i in sizes]
        plt.bar(x_axis, y_axis, width=.05, alpha=.25, label=sort.__name__)
    plt.legend()
    plt.title("Run Time of String Search Algorithms")
    plt.xlabel("Text and Pattern Sizes")
    plt.ylabel("Time for 100 Trials (ms)")
    plt.savefig("search_graph_bar.png")
    plt.show()


def main():
    trials = 100
    sizes = [(1000, 10), (2000, 20), (3000, 30), (4000, 40), (5000, 50)]
    dict_searches = {}
    searches = [native_search, brute_force, knuth_morris_pratt, rabin_karp]
    for search in searches:
        dict_searches[search.__name__] = {}

    for size in sizes:
        for search in searches:
            dict_searches[search.__name__][size] = 0
        for trial in range(1, trials):
            text = random_text(size[0])
            pattern = random_pattern(size[1], text)
            index_check = native_search(pattern, text)
            for search in searches:
                start_time = time.time()
                index = search(pattern, text)
                end_time = time.time()
                net_time = end_time - start_time
                dict_searches[search.__name__][size] += 1000 * net_time
                if index != index_check:
                    print("Error in", search.__name__)
                # print(search.__name__, "index:", index, "index check:", index_check )

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    df = pd.DataFrame.from_dict(dict_searches).T
    print(df)
    with open("console_output.txt", "w") as external_file:
        print(df, file=external_file)
        external_file.close()

    plot_times_bar_graph(dict_searches, sizes, searches)


if __name__ == '__main__':
    main()