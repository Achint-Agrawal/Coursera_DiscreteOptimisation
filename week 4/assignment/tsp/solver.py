#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import sys
import random
import numpy as np

MAX_TRIALS = 1000
MAX_SEARCHES = 100

Point = namedtuple("Point", ['x', 'y'])
points = []
nodeCount = 0
d = [[]]

def buildDistanceMatrix():
    def length(point1, point2):
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    global d
    d = np.asarray([[length(points[i], points[j]) for i in range(nodeCount)] for j in range(nodeCount)])

def twoLargest(a):      #returns index of two largest numbers in a
    c = a
    ma1 = max(c)
    x = np.where(a == ma1)
    c = np.delete(c, x)
    ma2 = max(c)
    y = np.where(a == ma2)
    return x,y

def swapEdges(s, i, j):
    if(i>j):
        s = np.append(s, s[0:i])
        s = s[i:]
        return swapEdges(s, 0, nodeCount - i + j)
    # print(s)
    return s

def changeInF(s, i, j):
    if(i==nodeCount):
        i = 0
    if(j == nodeCount):
        j = 0
    initial = d[s[i-1]][s[i]] + d[s[j-1]][s[j]]
    final = d[s[i-1]][s[j-1]] + d[s[i]][s[j]]
    return final-initial

def findDist(s):
    # print(d, s)
    dist = np.array([d[s[i]][s[i+1]] for i in range(nodeCount-1)])
    dist = np.append(dist, d[s[nodeCount-1]][s[0]])
    return dist

class f:
    def distanceSum(s):
        obj = d[s[-1]][s[0]]
        # print(s)
        for index in range(0, nodeCount-1):
            obj += d[s[index]][s[index+1]]
        return obj

class N:
    def twoOpt_random(s):
        x, y = random.sample(range(0, nodeCount), 2)
    
    def twoOpt_biggest2edges(s):
        # print('s=', s)
        dist = findDist(s)
        # print('dist = ', dist)
        x, y = twoLargest(dist)
        x =x[0]
        y = y[0]
        if(len(x) < 2):
            x = np.append(x, y)
        # print('x = ', x)
        Ns = np.empty((0, nodeCount), dtype=int)
        fn = np.array([])
        for i in range(len(x)):
            for j in range(i+2, len(x)):
                # print(x[i], x[j])
                n = swapEdges(s, i, j)
                # print('n=', n)
                fn = np.append(fn, f.distanceSum(s) + changeInF(s, x[i], x[j]))
                Ns = np.append(Ns, np.expand_dims(n, axis = 0), axis = 0)
        # print(Ns, fn)
        return Ns,fn

    def twoOpt_all(s):
        # print('s=', s)
        x = np.array(range(nodeCount))
        Ns = np.empty((0, nodeCount), dtype=int)
        fn = np.array([])
        for i in range(len(x)):
            for j in range(i+2, len(x)):
                # print(x[i], x[j])
                n = swapEdges(s, x[i], x[j])
                # print('n=', n)
                fn = np.append(fn, f.distanceSum(s) + changeInF(s, x[i], x[j]))
                Ns = np.append(Ns, np.expand_dims(n, axis = 0), axis = 0)
        # print(Ns, fn)
        return Ns,fn

class L:
    def greedy(N, s):
        # legal = np.empty((0, nodeCount))
        fs = f.distanceSum(s)
        # print(N, s)
        Ns, fs_ = N
        # for i in range(Ns.shape[0]):
        #     if(Ns[i][0] < fs):
        #         legal = np.append(legal, np.expand_dims(Ns[i][1], axis = 0), axis = 0)
        # return legal

        x = np.where(fs_ <= fs)
        # print('L', Ns[x], fs_[x])
        return Ns[x], fs_[x]

class S:
    def random(L, s):
        Ls, fs = L
        if(Ls.shape[0] == 0):
            return None
        index = np.random.randint(0, Ls.shape[0])
        return Ls[index], fs[index]

class InitialSolution:
    def random():
        return(np.random.permutation(nodeCount))

class Search:
    def local(f, N, L, S):
        s = InitialSolution.random()
        # s = np.array([3, 2, 1, 0, 4])
        # print(s)
        s_best = s
        fs_best = f(s_best)
        for k in range(MAX_TRIALS):
            s = S(L(N(s),s),s)
            if(s == None):
                break
            s, fs = s
            # print(s, fs)
            if(fs<fs_best):
                fs_best = fs
                s_best = S
        return s_best, fs_best

    def iteratedLocal(fd, N, L, S):
        s, f = Search.local(fd, N, L, S)
        s_best, f_best = s, f
        for i in range(MAX_SEARCHES):
            s, f = Search.local(fd, N, L, S)
            # print(f)
            if(f<f_best):
                f_best = f
                s_best = s
        return s_best, f_best
    

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # print('Input- \n', input_data)

    # parse the input
    lines = input_data.split('\n')

    global nodeCount
    nodeCount = int(lines[0])

    global points
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))
    buildDistanceMatrix()

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    # solution = np.array(range(0, nodeCount))

    # calculate the length of the tour
    # obj = f.distanceSum(solution)

    # solution, obj = Search.local(f.distanceSum, N.twoOpt_biggest2edges, L.greedy, S.random)
    solution, obj = Search.iteratedLocal(f.distanceSum, N.twoOpt_biggest2edges, L.greedy, S.random)
    # solution, obj = Search.iteratedLocal(f.distanceSum, N.twoOpt_all, L.greedy, S.random)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')
