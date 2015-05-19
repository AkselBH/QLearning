__author__ = 'Stubborn'

  #!/usr/bin/env python

from numpy import *
from matplotlib import pyplot as plt

from pybrain.rl.environments.mazes import Maze
from pybrain.rl.environments.mazes import MDPMazeTask
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q, ActionValueTable
from pybrain.rl.experiments import Experiment

envmatrix = array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 1, 1, 1, 0, 1],
                    [1, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 0, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1]])

environment = Maze(envmatrix, (7, 7))

task = MDPMazeTask(environment)

table = ActionValueTable(81, 4)
table.initialize(2.)

agent = LearningAgent(table, Q())

experiment = Experiment(task, agent)

plt.ion()
plt.hot()

x = raw_input('Want to start?')
yes = "yes"
no = "no"
if x == yes:

    for i in range(1000):
        experiment.doInteractions(100)
        agent.learn()
        agent.reset()

        plt.pcolor(table.params.reshape(81,4).max(axis=1).reshape(9, 9))
        plt.gcf().canvas.draw()

else:
    print "YOLO"