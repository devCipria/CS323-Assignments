# CSCI 323
# Winter 2022
# Assignment 2 (Sorting Algorithms)
# David Cipriati


import math
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def random_list(range_max, size):
    # Gives you a random array of size 'size'
    arr = [random.randint(1, range_max) for i in range(size)]
    return arr


# Python's native search
def native_sort(arr):
    arr.sort()


# https://www.geeksforgeeks.org/bubble-sort/
def bubble_sort(arr):
    n = len(arr)

    # Traverse through all array elements
    for i in range(n):

        # Last i elements are already in place
        for j in range(0, n-i-1):

            # traverse the array from 0 to n-i-1. Swap if the element found is greater than the next element
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]


# https://www.geeksforgeeks.org/selection-sort/
def selection_sort(arr):
    n = len(arr)
    for i in range(n):

        # Find the minimum element in remaining unsorted array
        min_idx = i
        for j in range(i + 1, n):
            if arr[min_idx] > arr[j]:
                min_idx = j

        # Swap the found minimum element with the first element
        arr[i], arr[min_idx] = arr[min_idx], arr[i]


# https://www.geeksforgeeks.org/insertion-sort/
def insertion_sort(arr):

    # Traverse through 1 to len(arr)
    for i in range(1, len(arr)):

        key = arr[i]

        # Move elements of arr[0..i-1], that are greater than key, to one position ahead of their current position
        j = i-1
        while j >= 0 and key < arr[j] :
            arr[j + 1] = arr[j]
            j -= 1
            arr[j + 1] = key


# https://www.geeksforgeeks.org/cocktail-sort/
def cocktail_sort(arr):
    n = len(arr)
    swapped = True
    start = 0
    end = n - 1
    while swapped:

        # reset the swapped flag on entering the loop, because it might be true from a previous iteration.
        swapped = False

        # loop from left to right same as the bubble sort
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True

        # if nothing moved, then array is sorted.
        if not swapped:
            break

        # otherwise, reset the swapped flag so that it can be used in the next stage
        swapped = False

        # move the end point back by one, because item at the end is in its rightful spot
        end = end - 1

        # from right to left, doing the same comparison as in the previous stage
        for i in range(end - 1, start - 1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True

        # inc. the starting point, because the last stage would have moved the next smallest num to its rightful spot
        start = start + 1


# https://www.geeksforgeeks.org/shellsort/
def shell_sort(arr):
    gap = len(arr) // 2  # initialize the gap

    while gap > 0:
        i = 0
        j = gap

        # check the array in from left to right till the last possible index of j
        while j < len(arr):

            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]

            i += 1
            j += 1

            # now, we look back from ith index to the left we swap the values which are not in the right order.
            k = i
            while k - gap > -1:

                if arr[k - gap] > arr[k]:
                    arr[k - gap], arr[k] = arr[k], arr[k - gap]
                k -= 1

        gap //= 2


# https://www.geeksforgeeks.org/merge-sort/
def merge_sort(arr):
    if len(arr) > 1:

        # Finding the mid of the array
        mid = len(arr)//2

        # Dividing the array elements
        left = arr[:mid]

        # into 2 halves
        right = arr[mid:]

        # Sorting the first half
        merge_sort(left)

        # Sorting the second half
        merge_sort(right)

        i = j = k = 0

        # Copy data to temp arrays L[] and R[]
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1


# partition function of quick sort
def partition(start, end, array):
    # Initializing pivot's index to start
    pivot_index = start
    pivot = array[pivot_index]

    # This loop runs till start pointer crosses end pointer, and then swaps the pivot with element on end pointer
    while start < end:

        # Increment the start pointer till it finds an element greater than  pivot
        while start < len(array) and array[start] <= pivot:
            start += 1

        # Decrement the end pointer till it finds an element less than pivot
        while array[end] > pivot:
            end -= 1

        # If start and end have not crossed each other, swap the numbers on start and end
        if start < end:
            array[start], array[end] = array[end], array[start]

    # Swap pivot element with element on end pointer. Moving the pivot in its correct sorted place.
    array[end], array[pivot_index] = array[pivot_index], array[end]

    # Returning end pointer to divide the array into 2
    return end


def quick_sort_recursive(start, end, array):
    if start < end:
        # p is partitioning index, array[p] is at right place
        p = partition(start, end, array)

        # Sort elements before partition and after partition
        quick_sort_recursive(start, p - 1, array)
        quick_sort_recursive(p + 1, end, array)


# https://www.geeksforgeeks.org/quick-sort/
def quick_sort(arr):
    quick_sort_recursive(0, len(arr) - 1, arr)


# heapify function of heap sort
def heapify(arr, n, i):
    largest = i  # Initialize largest as root
    left = 2 * i + 1     # left = 2*i + 1
    right = 2 * i + 2     # right = 2*i + 2

    # See if left child of root exists and is greater than root
    if left < n and arr[largest] < arr[left]:
        largest = left

    # See if right child of root exists and is greater than root
    if right < n and arr[largest] < arr[right]:
        largest = right

    # Change root, if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # swap

        # Heapify the root.
        heapify(arr, n, largest)


# https://www.geeksforgeeks.org/heap-sort/
def heap_sort(arr):
    n = len(arr)

    # Build a maxheap.
    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)

    # One by one extract elements
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # swap
        heapify(arr, i, 0)


# https://www.geeksforgeeks.org/counting-sort/
def counting_sort(arr):
    max_element = int(max(arr))
    min_element = int(min(arr))
    range_of_elements = max_element - min_element + 1
    # Create a count array to store count of individual elements and initialize count array as 0
    count_arr = [0 for _ in range(range_of_elements)]
    output_arr = [0 for _ in range(len(arr))]

    # Store count of each character
    for i in range(0, len(arr)):
        count_arr[arr[i]-min_element] += 1

    # Change count_arr[i] so that count_arr[i] now contains actual position of this element in output array
    for i in range(1, len(count_arr)):
        count_arr[i] += count_arr[i-1]

    # Build the output character array
    for i in range(len(arr)-1, -1, -1):
        output_arr[count_arr[arr[i] - min_element] - 1] = arr[i]
        count_arr[arr[i] - min_element] -= 1

    # Copy the output array to arr, so that arr now contains sorted characters
    for i in range(0, len(arr)):
        arr[i] = output_arr[i]

    return arr


def counting_sort_radix(arr, exp1):
    n = len(arr)

    # The output array elements that will have sorted arr
    output = [0] * n

    # initialize count array as 0
    count = [0] * 10

    # Store count of occurrences in count[]
    for i in range(0, n):
        index = arr[i] // exp1
        count[index % 10] += 1

    # Change count[i] so that count[i] now contains actual position of this digit in output array
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build the output array
    i = n - 1
    while i >= 0:
        index = arr[i] // exp1
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1

    # Copying the output array to arr[], so that arr now contains sorted numbers
    i = 0
    for i in range(0, len(arr)):
        arr[i] = output[i]


# https://www.geeksforgeeks.org/radix-sort/
# Method to do Radix Sort
def radix_sort(arr):
    # Find the maximum number to know number of digits
    max1 = max(arr)

    # Do counting sort for every digit. Note that instead of passing digit number, exp is passed. exp is 10^i
    # where i is current digit number
    exp = 1
    while max1 / exp > 1:
        counting_sort_radix(arr, exp)
        exp *= 10


# https://www.educative.io/edpresso/what-is-bucket-sort
def bucket_sort(arr):
    n = len(arr)
    # create list that will hold all the buckets (also lists)
    bucket_list = []
    bucket_number = n
    while bucket_number > 0:
        new_bucket = []
        bucket_list.append(new_bucket)
        bucket_number = bucket_number - 1

    maximum = max(arr)
    minimum = min(arr)

    # the numerical range that each bucket can hold
    element_range = (maximum - minimum) // n+1

    # place elements in their respective buckets
    for index in range(len(arr)):
        number = arr[index]
        difference = number - minimum
        bucket_number = int(difference // element_range)
        bucket_list[bucket_number].append(number)

    # sort all the buckets and merge them
    sorted_list = []
    for bucket in range(len(bucket_list)):
        current = bucket_list[bucket]
        current.sort()
        if len(current) != 0:
            sorted_list.extend(current)

    return sorted_list


def is_sorted(arr):
    if arr != sorted(arr):
        print("Error: The array is not sorted!")


def plot_times_bar_graph(dict_sorts, sizes, sorts):
    search_num = 0
    plt.xticks([j for j in range(len(sizes))], [str(size) for size in sizes])
    for sort in sorts:
        search_num += 1
        d = dict_sorts[sort.__name__]
        x_axis = [j + 0.05 * search_num for j in range(len(sizes))]
        y_axis = [d[i] for i in sizes]
        plt.bar(x_axis, y_axis, width=.05, alpha=.25, label=sort.__name__)
    plt.legend()
    plt.title("Run Time of Sort Algorithms")
    plt.xlabel("Number of Elements")
    plt.ylabel("Time for 100 Trials (ms)")
    plt.savefig("search_graph_bar.png")
    plt.show()


def main():
    trials = 5
    sizes = [100, 250, 500, 750, 1000]
    dict_sorts = {}
    sorts = [native_sort, bubble_sort, selection_sort, insertion_sort, cocktail_sort, shell_sort, merge_sort, quick_sort,
             heap_sort, counting_sort, radix_sort, bucket_sort]
    for sort in sorts:
        dict_sorts[sort.__name__] = {}

    for size in sizes:
        for sort in sorts:
            dict_sorts[sort.__name__][size] = 0
        for trial in range(1, trials):
            arr = random_list(9999, size)
            for sort in sorts:
                arr_copy = arr.copy()
                if sort == bucket_sort:
                    start_time = time.time()
                    arr_copy = sort(arr_copy)
                    end_time = time.time()
                    net_time = end_time - start_time
                else:
                    start_time = time.time()
                    sort(arr_copy)
                    end_time = time.time()
                    net_time = end_time - start_time

                is_sorted(arr_copy)
                dict_sorts[sort.__name__][size] += 1000 * net_time

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    df = pd.DataFrame.from_dict(dict_sorts).T
    print(df)
    with open("console_output.txt", "w") as external_file:
        print(df, file=external_file)
        external_file.close()

    plot_times_bar_graph(dict_sorts, sizes, sorts)


if __name__ == '__main__':
    main()