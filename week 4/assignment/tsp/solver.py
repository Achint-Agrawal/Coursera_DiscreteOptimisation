#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import sys
import random
import numpy as np

Point = namedtuple("Point", ['x', 'y'])
points = []
nodeCount = 0
# d = [[]]
MAX_SEARCHES = MAX_TRIALS = 0


def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def d(i, j):
    return length(points[i], points[j])

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
    initial = d(s[i+1], s[i]) + d(s[j+1], s[j])
    final = d(s[i], s[j]) + d(s[i+1],s[j+1])
    return final-initial

def findDist(s):
    dist = np.array([d(s[i], s[i+1]) for i in range(nodeCount-1)])
    dist = np.append(dist, d(s[nodeCount-1], s[0]))
    return dist

class f:
    def distanceSum(s):
        obj = d(s[-1],s[0])
        for index in range(0, nodeCount-1):
            obj += d(s[index], s[index+1])
        return obj


def metropolisDecision(fs, fn, t):
    probability = math.exp((fs-fn)/t)
    return random.random() < probability

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
                if(abs(x[i] - x[j]) != 1 and abs(x[i] - x[j]) != nodeCount-1):
                    # s2 = np.copy(s)
                    # n = swapEdges(s2, x[i], x[j])
                    
                    fn = np.append(fn, f.distanceSum(s) + changeInF(s, x[i], x[j]))
                    # Ns = np.append(Ns, np.expand_dims(n, axis = 0), axis = 0)
                    Ns = np.append(Ns, np.array([[x[i], x[j]]]), axis = 0)
        return Ns,fn

    def twoOpt_top_k(s):
        # if(len(x) < 2):
        dist = findDist(s)
        k = min(nodeCount, 25)
        x = np.argpartition(dist, -k)[-k:]
        # x = np.array(range(0, k))

        Ns = np.empty((0, 2), dtype=int)
        fn = np.array([])
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                if(abs(x[i] - x[j]) != 1 and abs(x[i] - x[j]) != nodeCount-1):
                    # s2 = np.copy(s)
                    # n = swapEdges(s2, x[i], x[j])
                    
                    fn = np.append(fn, f.distanceSum(s) + changeInF(s, x[i], x[j]))
                    # Ns = np.append(Ns, np.expand_dims(n, axis = 0), axis = 0)
                    Ns = np.append(Ns, np.array([[x[i], x[j]]]), axis = 0)
        return Ns,fn

    def twoOpt_all(s):
        x = np.array(range(0, nodeCount))

        Ns = np.empty((0, 2), dtype=int)
        fn = np.array([])
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                if(abs(x[i] - x[j]) != 1 and abs(x[i] - x[j]) != nodeCount-1):
                    # s2 = np.copy(s)
                    # n = swapEdges(s2, x[i], x[j])
                    
                    fn = np.append(fn, f.distanceSum(s) + changeInF(s, x[i], x[j]))
                    # Ns = np.append(Ns, np.expand_dims(n, axis = 0), axis = 0)
                    Ns = np.append(Ns, np.array([[x[i], x[j]]]), axis = 0)
        return Ns,fn

    def twoOpt_random_k(s):
        k = min(nodeCount, 50)
        x = random.sample(range(0, nodeCount), k)
        # x = np.array(range(0, k))

        Ns = np.empty((0, 2), dtype=int)
        fn = np.array([])
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                if(abs(x[i] - x[j]) != 1 and abs(x[i] - x[j]) != nodeCount-1):
                    # s2 = np.copy(s)
                    # n = swapEdges(s2, x[i], x[j])
                    
                    fn = np.append(fn, f.distanceSum(s) + changeInF(s, x[i], x[j]))
                    # Ns = np.append(Ns, np.expand_dims(n, axis = 0), axis = 0)
                    Ns = np.append(Ns, np.array([[x[i], x[j]]]), axis = 0)
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
    
    def best(L, s):
        Ls, fs = L
        # print(Ls, fs)
        if(Ls.shape[0] == 0):
            return None
        
        index = np.where(fs == np.min(fs))
        Ls = Ls[index]
        fs = fs[index]
        index = np.random.randint(0, Ls.shape[0])
        sol = s
        sol = swapEdges(sol, Ls[index][0], Ls[index][1])
        return sol, fs[index]
    
    def metropolis(t, L, s):
        Ls, fs = L
        if(Ls.shape[0] == 0):
            return None
        index = np.random.randint(0, Ls.shape[0])   
        sol = s
        n = swapEdges(sol, Ls[index][0], Ls[index][1])
        fn = fs[index]
        fi = f.distanceSum(s)
        if(fn<fi):
            return n, fn
        else:
            if(metropolisDecision(fi, fn, t)):
                return n, fn
            else:
                return s, fi


class InitialSolution:
    def random():
        return(np.random.permutation(nodeCount))
    def default():
        return(np.array(range(nodeCount)))

class Search:
    def local(f, N, L, S, initialSolution):
        # s = np.array([4, 0, 2, 1, 3])
        # s= InitialSolution.random()
        s = initialSolution()
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

    def iteratedLocal(fd, N, L, S, initialSolution):
        s, f = Search.local(fd, N, L, S, initialSolution)
        s_best, f_best = s, f
        for i in range(MAX_SEARCHES):
            s, f = Search.local(fd, N, L, S, initialSolution)
            print(f)
            if(f<f_best):
                f_best = f
                s_best = s
        return s_best, f_best

    def iteratedLocal_objective(fd, N, L, S, initialSolution):
        s, f = Search.local(fd, N, L, S, initialSolution)
        s_best, f_best = s, f
        print(f)
        print(f_best, getObjective())
        while(f_best > getObjective()):
            s, f = Search.local(fd, N, L, S, initialSolution)
            print(f)
            if(f<f_best):
                f_best = f
                s_best = s
        return s_best, f_best


def initializeObjectives():
    global points, MAX_SEARCHES, MAX_TRIALS, nodeCount
    points = []
    MAX_TRIALS = 1000
    if(nodeCount>10000):
        MAX_SEARCHES = 10
    else:
        MAX_SEARCHES = 100

def getObjective():
    if(nodeCount == 51):
        return 482 
    if(nodeCount == 100):
        return 23433  
    if(nodeCount == 200):
        return 35985 
    if(nodeCount == 574):
        return 40000  
    if(nodeCount == 1889):
        return 378069  
    if(nodeCount == 33810):
        return 78478868 
    return 1e9
     
def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # print(input_data)
    # f = open('input2.txt', 'w')
    # f.write(input_data)
    # f.close()

    # parse the input
    
    global nodeCount
    nodeCount = 0

    lines = input_data.split('\n')

    nodeCount = int(lines[0])
    initializeObjectives()
    

    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))
    # buildDistanceMatrix()

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    # solution = np.array(range(0, nodeCount))

    # calculate the length of the tour
    # obj = f.distanceSum(solution)

    print(nodeCount)

    # solution, obj = Search.local(f.distanceSum, N.twoOpt_biggest2edges, L.greedy, S.random, InitialSolution.default)
    # solution, obj = Search.iteratedLocal(f.distanceSum, N.twoOpt_biggest2edges, L.greedy, S.random, InitialSolution.random)
    # solution, obj = Search.iteratedLocal(f.distanceSum, N.twoOpt_top_k, L.greedy, S.random, InitialSolution.random)
    # solution, obj = Search.iteratedLocal(f.distanceSum, N.twoOpt_random_k, L.greedy, S.best, InitialSolution.random)
    # solution, obj = Search.iteratedLocal(f.distanceSum, N.twoOpt_all, L.greedy, S.best, InitialSolution.random)
    # solution, obj = Search.iteratedLocal(f.distanceSum, N.twoOpt_biggest2edges, L.greedy, S.best, InitialSolution.random)

    if(nodeCount < 500):
        solution, obj = Search.iteratedLocal_objective(f.distanceSum, N.twoOpt_random_k, L.greedy, S.best, InitialSolution.random)
    else:
        solution, obj = Search.iteratedLocal(f.distanceSum, N.twoOpt_biggest2edges, L.greedy, S.best, InitialSolution.random)



    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    # output_data = 0
    
    
    output_data += ' '.join(map(str, solution))

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
