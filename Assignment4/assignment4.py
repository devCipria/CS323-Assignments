# CSCI 323
# Winter 2022
# Assignment 4 (Graph Algorithms)
# David Cipriati

import math
import random
import sys
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join


def read_graph(file_name):
    with open(file_name, 'r') as file:
        graph = []
        lines = file.readlines()
        for line in lines:
            costs = line.split(' ')
            row = []
            for cost in costs:
                row.append(int(cost))
            graph.append(row)
        return graph


def desc_graph(graph):
    num_vertices = len(graph)
    message = ''
    message += 'Number of vertices = ' + str(num_vertices) + '\n'
    non_zero = 0
    for i in range(num_vertices):
        for j in range(num_vertices):
            if graph[i][j] > 0:
                non_zero += 1
    num_edges = int(non_zero / 2)
    message += 'Number of edges = ' + str(num_edges) + '\n'
    message += 'Symmetric = ' + str(is_symmetric(graph)) + '\n'
    return message


def is_symmetric(graph):
    num_vertices = len(graph)
    for i in range(num_vertices):
        for j in range(num_vertices):
            if graph[i][j] != graph[j][i]:
                return False
    return True


def print_graph(graph, sep=' '):
    str_graph = ''
    for row in range(len(graph)):
        str_graph += sep.join([str(c) for c in graph[row]]) + '\n'
    return str_graph


def analyze_graph(file_name):
    graph = read_graph(file_name)
    output_file_name = file_name[0:-4 + len(file_name)] + '_report.txt'
    with open(output_file_name, 'w') as output_file:
        output_file.write('Analysis of graph: ' + file_name + '\n\n')
        str_graph = print_graph(graph)
        output_file.write(str_graph + '\n')
        graph_description = desc_graph(graph)
        output_file.write(graph_description + '\n')

        start_time = time.time()
        dfs_traversal = dfs(graph)
        end_time = time.time()
        dfs_net_time = (end_time - start_time) * 1000

        start_time = time.time()
        bfs_traversal = bfs(graph)
        end_time = time.time()
        bfs_net_time = (end_time - start_time) * 1000

        start_time = time.time()
        prim_mst_print = prim_mst(graph)
        end_time = time.time()
        prim_net_time = (end_time - start_time) * 1000

        start_time = time.time()
        kruskal_mst_print = kruskal_mst(graph)
        end_time = time.time()
        kruskal_net_time = (end_time - start_time) * 1000

        start_time = time.time()
        dijkstra_print = dijkstra_apsp(graph)
        end_time = time.time()
        dijkstra_net_time = (end_time - start_time) * 1000

        start_time = time.time()
        floyd_print = floyd_apsp(graph)
        end_time = time.time()
        floyd_net_time = (end_time - start_time) * 1000

        output_file.write('dfs traversal: ' + str(dfs_traversal) + '\nTime in milliseconds: ' + str(dfs_net_time) + "\n\n")
        output_file.write('bfs traversal: ' + str(bfs_traversal) + '\nTime in milliseconds: ' + str(bfs_net_time) + "\n\n")
        output_file.write("\nPrim's MST: " + prim_mst_print + 'Time in milliseconds: ' + str(prim_net_time) + "\n\n")
        output_file.write("\nKruskal's MST: " + kruskal_mst_print + 'Time in milliseconds: ' + str(kruskal_net_time) + "\n\n")
        output_file.write("Dijkstra's APSP:\n" + dijkstra_print + '\nTime in milliseconds: ' + str(dijkstra_net_time) + "\n\n")
        output_file.write("Floyd's APSP:\n" + floyd_print + '\n\nTime in milliseconds: ' + str(floyd_net_time))

        print('dfs traversal: ' + str(dfs_traversal) + '\nTime in milliseconds: ' + str(dfs_net_time) + "\n\n")
        print('bfs traversal: ' + str(bfs_traversal) + '\nTime in milliseconds: ' + str(bfs_net_time) + "\n\n")
        print("\nPrim's MST: " + prim_mst_print + 'Time in milliseconds: ' + str(prim_net_time) + "\n\n")
        print("\nKruskal's MST: " + kruskal_mst_print + 'Time in milliseconds: ' + str(kruskal_net_time) + "\n\n")
        print("Dijkstra's APSP:\n" + dijkstra_print + '\nTime in milliseconds: ' + str(dijkstra_net_time) + "\n\n")
        print("Floyd's APSP:\n" + floyd_print + '\n\nTime in milliseconds: ' + str(floyd_net_time))


def dfs_util(graph, v, visited):
    visited.append(v)
    for col in range(len(graph[v])):
        if graph[v][col] > 0 and col not in visited:
            dfs_util(graph, col, visited)


# https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/
def dfs(graph):
    visited = []
    dfs_util(graph, 0, visited)
    return visited


# https://www.geeksforgeeks.org/implementation-of-bfs-using-adjacency-matrix/
def bfs(graph):
    # Visited vector to so that a vertex is not visited more than once
    # Initializing the vector to false as no vertex is visited at the beginning
    start = 0
    visited = [False] * len(graph)
    bfs_traversal_path = []
    bfs_traversal_path.append(start)

    q = [start]

    # Set source as visited
    visited[start] = True

    while q:
        vis = q[0]

        q.pop(0)

        # For every adjacent vertex to the current vertex
        for i in range(len(graph)):
            if graph[vis][i] > 0 and (not visited[i]):
                q.append(i)
                visited[i] = True
                bfs_traversal_path.append(i)
    return bfs_traversal_path


def print_mst(graph, parent):
    title = "\nEdge \tWeight"
    edges_weights = title + "\n"
    total_cost = 0

    for i in range(1, len(graph)):
        edges_weights += (str(parent[i]) + " - " + str(i) + " \t " + str(graph[i][parent[i]]) + "\n")
        total_cost += graph[i][parent[i]]
    edges_weights += "Total Weight of Prim's MST:" + str(total_cost) + "\n"
    return edges_weights


def min_key(graph, key, mst_set):
    # Initialize min value
    min_value = sys.maxsize

    for v in range(len(graph)):
        if key[v] < min_value and mst_set[v] == False:
            min_value = key[v]
            min_index = v

    return min_index


# https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
def prim_mst(graph):
    # Key values used to pick minimum weight edge in cut
    key = [sys.maxsize] * len(graph)
    parent = [None] * len(graph)  # Array to store constructed MST
    # Make key 0 so that this vertex is picked as first vertex
    key[0] = 0
    mst_set = [False] * len(graph)

    parent[0] = -1  # First node is always the root of

    for cout in range(len(graph)):
        u = min_key(graph, key, mst_set)
        mst_set[u] = True

        for v in range(len(graph)):
            if graph[u][v] > 0 and mst_set[v] == False and key[v] > graph[u][v]:
                key[v] = graph[u][v]
                parent[v] = u

    prim_output = print_mst(graph, parent)
    return prim_output


def set_to_inf(graph):
    inf_graph = graph
    infinity = float('inf')
    for i in range (len(inf_graph)):
        for j in range (len(inf_graph)):
            if inf_graph[i][j] == 0:
                inf_graph[i][j] = infinity
    return inf_graph


def find(parent, i):
    while parent[i] != i:
        i = parent[i]
    return i


def union(parent, i, j):
    a = find(parent, i)
    b = find(parent, j)
    parent[a] = b


# https://www.geeksforgeeks.org/kruskals-algorithm-simple-implementation-for-adjacency-matrix/
def kruskal_mst(graph):
    i_graph = set_to_inf(graph)
    min_cost = 0  # Cost of min MST
    parent = [i for i in range(len(i_graph))]
    title = "\nEdge \tWeight"
    edges_weights = title + "\n"

    # Initialize sets of disjoint sets
    for i in range(len(i_graph)):
        parent[i] = i

    # Include minimum weight edges one by one
    edge_count = 0
    while edge_count < (len(i_graph) - 1):
        minimum = float('inf')
        a = -1
        b = -1
        for i in range(len(i_graph)):
            for j in range(len(i_graph)):
                if find(parent, i) != find(parent, j) and i_graph[i][j] < minimum:
                    minimum = i_graph[i][j]
                    a = i
                    b = j
        union(parent, a, b)
        edges_weights += (str(a) + " - " + str(b) + " \t " + str(minimum) + "\n")
        edge_count += 1
        min_cost += minimum

    edges_weights += "Total Weight of Kruskal's MST:" + str(min_cost) + "\n"
    return edges_weights


# from class slides
# https://www.geeksforgeeks.org/floyd-warshall-algorithm-dp-16/
def floyd_apsp(graph):
    i_graph = set_to_inf(graph)
    num_v = len(i_graph)
    maxsize = float('inf')
    dist = [[maxsize for _ in range(num_v)]for _ in range(num_v)]
    pred = [[0 for _ in range(num_v)]for _ in range(num_v)]

    for i in range(num_v):
        for j in range(num_v):
            dist[i][j] = i_graph[i][j]  # path of length 1, i.e. just the edge
            pred[i][j] = i	 # predecessor will be vertex i
        dist[i][i] = 0  # no cost
        pred[i][i] = -1  # indicates end of path

    for k in range(num_v):
        for i in range(num_v):
            for j in range(num_v):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    pred[i][j] = pred[k][j]
                    dist[i][j] = dist[i][k] + dist[k][j]

    print_arrays = "Dist Array:\n" + str(dist) + "\nPred Array:\n" + str(pred)
    return print_arrays


def printSolution(graph, dist):
    print("Vertex tDistance from Source")
    for node in range(len(graph)):
        print(node, "t", dist[node])


def minDistance(graph, dist, sptSet):
    min = 1e7

    for v in range(len(graph)):
        if dist[v] < min and sptSet[v] == False:
            min = dist[v]
            min_index = v

    return min_index


# class slides
# https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/
def dijkstra_apsp(graph):
    INF = float('inf')
    length = len(graph)
    print_dist = ""
    print_pred = ""

    for src in range(length):
        dist = [INF] * length
        dist[src] = src
        pred = [0] * length
        visited = [False] * length

        for cout in range(length):
            u = minDistance(graph, dist, visited)
            visited[u] = True

            for v in range(length):
                if graph[u][v] > 0 and visited[v] == False and dist[v] > dist[u] + graph[u][v]:
                    dist[v] = dist[u] + graph[u][v]
                    pred[v] = u
        print_dist += str(dist) + "\n"
        print_pred += str(pred) + "\n"

    print_arrays = "Dist Array:\n" + print_dist + "\nPred Array:\n" + print_pred
    return print_arrays


def main():
    mypath = "C:\\Users\\trade\\PycharmProjects\\323\\Assignment4\\"
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for file in files:
        if file[0:5] == 'graph' and file.find('_report') < 0:
            analyze_graph(file)


if __name__ == '__main__':
    main()
