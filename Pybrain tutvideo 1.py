__author__ = 'Stubborn'


from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers.backprop import BackpropTrainer

D = SupervisedDataSet(2, 1)
# 2 imput --> 1 output

D.addSample([0,0], [0])
D.addSample([0,1], [1])
D.addSample([1,0], [1])
D.addSample([1,1], [0])
# 4 kombinationer av input och dess output "OR" funktion

N = buildNetwork(2, 4, 1)
# multilayer perception? med 1 gömt lager

T = BackpropTrainer(N, learningrate = 0.01, momentum = 0.99)
# momentum = reduced learningrate

print (('MSE before'), T.testOnData(D))
T.trainOnDataset(D, 1000)
print (('MSE after'), T.testOnData(D))