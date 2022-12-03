#!/usr/bin/env python3

import math
import random
import time
from enum import Enum
import pandas as pd
import numpy as np

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtWidgets import *
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

# TODO: Error checking on txt boxes
# TODO: Color strings


# Import in the code with the actual implementation
from TSPSolver import *
# from TSPSolver_complete import *
from TSPClasses import *


class Tester:

    def __init__(self):
        self.timeLimit = None
        self.diff = None
        self.size = None
        self.curSeed = None
        self.prunedStates = None
        self.totalStates = None
        self.maxQSize = None
        self.solvedIn = None
        self._solution = None
        self.tourCost = None
        self.diffDropDown = None
        self.view = None
        self.data_range = None
        self.numSolutions = None
        self._MAX_SEED = 1000
        self.algorithm = None

        self._scenario = None
        self.solver = TSPSolver(self.view)
        self.genParams = {'size': None, 'seed': None, 'diff': None}

        SCALE = 1.0
        self.data_range = {'x': [-1.5 * SCALE, 1.5 * SCALE], \
                           'y': [-SCALE, SCALE]}


    def newPoints(self):
        # TODO - ERROR CHECKING!!!!
        random.seed(self.curSeed)

        ptlist = []
        RANGE = self.data_range
        xr = self.data_range['x']
        yr = self.data_range['y']
        npoints = int(self.size)
        while len(ptlist) < npoints:
            x = random.uniform(0.0, 1.0)
            y = random.uniform(0.0, 1.0)
            if True:
                xval = xr[0] + (xr[1] - xr[0]) * x
                yval = yr[0] + (yr[1] - yr[0]) * y
                ptlist.append(QPointF(xval, yval))
        return ptlist

    def generateNetwork(self):
        points = self.newPoints()  # uses current rand seed
        diff = self.diff
        rand_seed = int(self.curSeed)
        self._scenario = Scenario(city_locations=points, difficulty=diff, rand_seed=rand_seed)

        self.genParams = {'size': self.size, 'seed': self.curSeed, 'diff': diff}

    def genRandSeed(self):
        self.curSeed = random.randint(0, self._MAX_SEED - 1)

    def returnParam(self):
        # ["size", "seed", "time", "cost", "max Q", "# soln", "tot States", "pruned"]
        return [self.size, self.curSeed, self.solvedIn, self.tourCost, self.maxQSize, self.numSolutions,
                self.totalStates, self.prunedStates]

    def solve(self, ncities=14, seed=1, randSeed=False, algo='branchAndBound', difficulty='Hard (Deterministic)',
              time_limit=60):

        if randSeed:
            self.genRandSeed()
        else:
            self.curSeed = seed
        self.size = ncities
        self.diff = difficulty
        self.timeLimit = time_limit


        self.algorithm = algo
        self.generateNetwork()
        self.solver.setupWithScenario(self._scenario)

        max_time = float(self.timeLimit)
        solve_func = 'self.solver.' + self.algorithm
        results = eval(solve_func)(time_allowance=max_time)
        if results:
            self.numSolutions = results['count']
            self.tourCost = results['cost']
            self.solvedIn = results['time']
            self._solution = results['soln']
            if 'max' in results.keys():
                self.maxQSize = results['max']
            if 'total' in results.keys():
                self.totalStates = results['total']
            if 'pruned' in results.keys():
                self.prunedStates = results['pruned']
        else:
            print('GOT NULL SOLUTION BACK!!')  # probably shouldn't ever use this...

    ALGORITHMS = [ \
        ('Default                            ', 'defaultRandomTour'), \
        ('Greedy', 'greedy'), \
        ('Branch and Bound', 'branchAndBound'), \
        ('Fancy', 'fancy') \
        ]  # whitespace hack to get longest to display correctly

class Algorithm(Enum):
    DEFAULT = 'defaultRandomTour'
    GREEDY = 'greedy'
    BnB = 'branchAndBound'
    FANCY = 'fancy'

class Difficulty(Enum):
    EASY = 'Easy'
    NORMAL = 'Normal'
    HARD = 'Hard'
    HARD_DET = 'Hard (Deterministic)'

if __name__ == '__main__':


    names = ["size", "seed", "time", "cost", "max Q", "# soln", "tot States", "pruned"]

    outputs = []

    test = Tester()

    # names = [0"size", 1"seed", 2"num Solutions", 3"tour Cost", 4"time", 5"max Q Size", 6"total States",
    # 7"pruned States"]
    inputs = [[15, 20, False, 'branchAndBound', 'Hard (Deterministic)', 60],
              [16, 902, False, 'branchAndBound', 'Hard (Deterministic)', 60],
              [10, 1, True, 'branchAndBound', 'Hard (Deterministic)', 60],
              [12, 1, True, 'branchAndBound', 'Hard (Deterministic)', 60],
              [13, 1, True, 'branchAndBound', 'Hard (Deterministic)', 60],
              [20, 1, True, 'branchAndBound', 'Hard (Deterministic)', 60],
              [25, 1, True, 'branchAndBound', 'Hard (Deterministic)', 60],
              [30, 1, True, 'branchAndBound', 'Hard (Deterministic)', 60],
              [40, 1, True, 'branchAndBound', 'Hard (Deterministic)', 60],
              [50, 1, True, 'branchAndBound', 'Hard (Deterministic)', 60]]

    outputs = []

    for input_list in inputs:
        test.solve(input_list[0], input_list[1], input_list[2], input_list[3], input_list[4], input_list[5])
        outputs.append(test.returnParam())

    table = pd.DataFrame(outputs, columns=names)
    pd.set_option("display.precision", 2)
    print(table)
    print(table.describe())

