#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    # value = 0
    # weight = 0
    # taken = [0]*len(items)

    # for item in items:
    #     print(item)
    #     if weight + item.weight <= capacity:
    #         taken[item.index] = 1
    #         value += item.value
    #         weight += item.weight



    # My solution...
    # print(item_count, capacity)
    # print(lines)
    # value = 0
    # weight = 0
    taken = [0]*len(items)

    dp = np.zeros([capacity+1, len(items) + 1], int)
    for budget in range(1,capacity + 1):
        for item in items:
            i = item.index + 1
            dp[budget][i] = dp[budget][i-1]
            if(item.weight <= budget):
                dp[budget][i] = max(dp[budget][i], item.value + dp[budget - item.weight][i-1])
    value = dp[capacity][len(items)]
    # print(dp)
    
    i = capacity
    for j in range(len(items), 1, -1):
        if(dp[i][j] != dp[i][j-1]):
            taken[j-1] = 1
            i -= items[j-1].weight

    

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

