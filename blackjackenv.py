__author__ = 'aksel'

from pybrain.rl.environments.environment import Environment
from scipy import zeros

class BlackjackEnv(Environment):
    """ A (terribly simplified) Blackjack game implementation of an environment. """

    # the number of action values the environment accepts
    indim = 2

    # the number of sensor values the environment produces
    outdim = 21

    def getSensors(self):
        """ the currently visible state of the world (the    observation may be stochastic - repeated calls returning different values)
            :rtype: by default, this is assumed to be a numpy array of doubles
        """
        hand_value = int(raw_input("Enter hand value: ")) - 1
        return [float(hand_value),]

    def performAction(self, action):
        """ perform an action on the world that changes it's internal state (maybe stochastically).
            :key action: an action that should be executed in the Environment.
            :type action: by default, this is assumed to be a numpy array of doubles
        """
        print "Action performed: ", action

    def reset(self):
        """ Most environments will implement this optional method that allows for reinitialization.
        """