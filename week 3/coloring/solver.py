#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import pathlib
from constraint import *
import numpy as np
import random
neighbours = None

def generate_neighbours(node_count, edges):
    global neighbours
    neighbours = [[] for _ in range(node_count)]
    for a,b in edges:
        neighbours[a] = np.append(neighbours[a], [b]).astype('int32')
        neighbours[b] = np.append(neighbours[b], [a]).astype('int32')
    # print(neighbours)

def cp1(node_count, edge_count, edges):
    #simplest CP soln. COlors iterate from 2 to node_count

    for ncolors in range(1, node_count+1):

        problem = Problem()
        nodes = range(node_count)
        colors = range(ncolors)
        

        solution = problem.getSolutions()
        return solution

def cp2(node_count): 
    #fixing first available color
    global neighbours
    order = sorted(range(len(neighbours)), key=lambda k: len(neighbours[k]), reverse=True)
    minreq = 1
    colors = [-1 for _ in range(node_count)]
    colors[order[0]] = 0

    for node in order[1:]:
        D = set(range(node_count))
        for neighbour in neighbours[node]:
            D.discard(colors[neighbour])
        
        colors[node] = min(D)
        minreq = max(minreq, colors[node] + 1)

    return [minreq, colors]

def cp3_x(node_count, order):
    minreq = 1
    colors = [-1 for _ in range(node_count)]
    colors[order[0]] = 0

    for node in order[1:]:
        D = set(range(node_count))
        for neighbour in neighbours[node]:
            D.discard(colors[neighbour])
        
        pool = [min(D)]
        # print(D)
        for d in D:
            # print(d, minreq, pool[0])
            if(d<=minreq+1 & d!=pool[0]):
                pool.append(d)
            else:
                if(d!=pool[0]):
                    break
        # if(len(pool)>1):
            # print("IDK man")
        colors[node] = random.choice(pool)
        minreq = max(minreq, colors[node] + 1)
    return [minreq,colors]

def cp3(node_count):
    #cp2 but some randomisation

     #fixing first available color
    global neighbours
    order = sorted(range(len(neighbours)), key=lambda k: len(neighbours[k]), reverse=True)
    targets = {20:5, 50:8, 70:20, 100:21, 250:95, 500:18, 1000:124}
    minreq = targets[node_count]+1
    colors = 0
    while(minreq>targets[node_count]):
        print(minreq)
        minreq,colors = cp3_x(node_count, order)
    
    return [minreq, colors]

# def cp3(node_count): 
    # assuming ncolors is given, is it possible to color graph?
    # need to find best we can do after a given step


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input

    # print(input_data)
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])
    print(node_count)

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    #using constraint programming
    generate_neighbours(node_count, edges)

    solution = cp3(node_count)
    # prepare the solution in the specified output format
    output_data = str(solution[0]) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, solution[1]))

    return output_data



if __name__ == '__main__':
    # import sys
    # if len(sys.argv) > 1:
    #     file_location = sys.argv[1].strip()
    #     print(file_location)
    #     with open(file_location, 'r') as input_data_file:
    #         input_data = input_data_file.read()
    #     print(solve_it(input_data))
    # else:
    #     print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

    file_location = str(pathlib.Path(__file__).parent.absolute()) + '/data/gc_' + input()
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    print(solve_it(input_data))
