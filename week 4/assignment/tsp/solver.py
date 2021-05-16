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
        i,j = j,i
    i = i+1
    j = j+1
    s[i:j] = np.flip(s[i:j])
    return s

def changeInF(s, i, j):

    s = np.append(s, s[0])
    initial = d[s[i+1]][s[i]] + d[s[j+1]][s[j]]
    final = d[s[i]][s[j]] + d[s[i+1]][s[j+1]]
    return final-initial

def findDist(s):
    dist = np.array([d[s[i]][s[i+1]] for i in range(nodeCount-1)])
    dist = np.append(dist, d[s[nodeCount-1]][s[0]])
    return dist

class f:
    def distanceSum(s):
        obj = d[s[-1]][s[0]]
        for index in range(0, nodeCount-1):
            obj += d[s[index]][s[index+1]]
        return obj

class N:
    def twoOpt_random(s):
        x, y = random.sample(range(0, nodeCount), 2)
    
    def twoOpt_biggest2edges(s):
        dist = findDist(s)
        x, y = twoLargest(dist)
        x =x[0]
        y = y[0]
        # if(len(x) < 2):
        x = np.append(x, y)
        Ns = np.empty((0, 2), dtype=int)
        fn = np.array([])
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                if(abs(x[i] - x[j]) != 1):
                    # s2 = np.copy(s)
                    # n = swapEdges(s2, x[i], x[j])
                    
                    fn = np.append(fn, f.distanceSum(s) + changeInF(s, x[i], x[j]))
                    # Ns = np.append(Ns, np.expand_dims(n, axis = 0), axis = 0)
                    Ns = np.append(Ns, np.array([[x[i], x[j]]]), axis = 0)
        return Ns,fn

    def twoOpt_all(s):
        x = np.array(range(nodeCount))
        Ns = np.empty((0, nodeCount), dtype=int)
        fn = np.array([])
        for i in range(len(x)):
            for j in range(i+2, len(x)):
                n = swapEdges(s, x[i], x[j])
                
                fn = np.append(fn, f.distanceSum(s) + changeInF(s, x[i], x[j]))
                Ns = np.append(Ns, np.expand_dims(n, axis = 0), axis = 0)
        return Ns,fn

class L:
    def greedy(N, s):
        fs = f.distanceSum(s)
        Ns, fs_ = N
        x = np.where(fs_ <= fs)
        return Ns[x], fs_[x]

class S:
    def random(L, s):
        Ls, fs = L
        if(Ls.shape[0] == 0):
            return None
        index = np.random.randint(0, Ls.shape[0])
        sol = s
        sol = swapEdges(sol, Ls[index][0], Ls[index][1])
        return sol, fs[index]

class InitialSolution:
    def random():
        return(np.random.permutation(nodeCount))

class Search:
    def local(f, N, L, S):
        # s = np.array([4, 0, 2, 1, 3])
        s= InitialSolution.random()
        s_best = s
        fs_best = f(s_best)
        for k in range(MAX_TRIALS):
            s = S(L(N(s),s),s)
            
            if(s == None):
                break
            s, fs = s
            if(fs<fs_best):
                fs_best = fs
                s_best = s
        return s_best, fs_best

    def iteratedLocal(fd, N, L, S):
        s, f = Search.local(fd, N, L, S)
        s_best, f_best = s, f
        for i in range(MAX_SEARCHES):
            s, f = Search.local(fd, N, L, S)
            print(f)
            if(f<f_best):
                f_best = f
                s_best = s
        return s_best, f_best
    

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # print(input_data)
    # f = open('input2.txt', 'w')
    # f.write(input_data)
    # f.close()

    # parse the input
    global points
    global nodeCount
    global d
    points = []
    nodeCount = 0
    d = [[]]

    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    
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
    # solution, obj = Search.iteratedLocal(f.distanceSum, N.twoOpt_biggest2edges, L.greedy, S.random)
    # solution, obj = Search.iteratedLocal(f.distanceSum, N.twoOpt_all, L.greedy, S.random)

    # prepare the solution in the specified output format
    # output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data = 0
    

    # output_data += ' '.join(map(str, solution))

    # print(obj, f.distanceSum(solution))

    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        
        print(solve_it(input_data))
        # solve_it(input_data)
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')
