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

envmatrix2 = array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                    [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                    [1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                    [1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                    [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                    [1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

environment = Maze(envmatrix2, (6, 7))

task = MDPMazeTask(environment)

table = ActionValueTable(324, 4)
table.initialize(3.)

agent = LearningAgent(table, Q())

experiment = Experiment(task, agent)

plt.ion()
plt.hot()
#plt.set_cmap() Fix bakgrundsfarg

x = raw_input('Want to start?')
yes = "yes"
no = "no"
if x == yes:

    for i in range(1000):
        experiment.doInteractions(100)
        agent.learn()
        agent.reset()

        plt.pcolor(table.params.reshape(324,4).max(axis=1).reshape(18, 18))
        plt.gcf().canvas.draw()

else:
    print "Wrong answer"