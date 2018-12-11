"""This is where you define your own agents for inclusion in the competition.

(This is the only file that you submit.)

Initially this is a copy of random_team.py and so defines a team comprised of
agents that just choose random actions.

Shows the minimal code that needs to be implemented to have a working team.

Champlain College CSI-480, Fall 2018
The following code was adapted by Joshua Auerbach (jauerbach@champlain.edu)
from the UC Berkeley Pacman Projects (see license and attribution below).

----------------------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""
import util
from capture_agents import CaptureAgent
import random


#################
# Team creation #
#################


def create_team(first_index, second_index, is_red,
                first='OffenceAgent', second='DefenceAgent'):
    """Create a team with agent indices first_index and second_index.

    This function returns a list of two agents that will form the
    team, initialized using first_index and second_index as their agent
    index numbers.  is_red is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    # The following line is an example only; feel free to change it.
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########


class BasicAgent(CaptureAgent):
    """A Dummy agent to serve as an example of the necessary agent structure.

    You should look at baseline_team.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def register_initial_state(self, game_state):
        """Handle initial setup of the agent to populate useful fields.

        Useful fields include things such as what team we're on.

        A distance_calculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.get_distance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """
        # Make sure you do not delete the following line. If you would like to
        # use Manhattan distances instead of maze distances in order to save
        # on initialization time, please take a look at
        # CaptureAgent.register_initial_state in capture_agents.py.
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        # Your initialization code goes here, if you need any.

    def choose_action(self, game_state):
        """Choose from avaliable actions randomly."""
        actions = game_state.get_legal_actions(self.index)

        # You should change this in your own agent.

        return random.choice(actions)

    def get_successor(self, game_state, action):
        return game_state.generate_successor(self.index, action)

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return None

    def get_weights(self, game_state, action):
        return {'successor_score': 1.0}


class OffenceAgent(BasicAgent):
    def get_features(self, game_state, action):
        super().get_features(game_state, action)

    def get_weights(self, game_state, action):
        super().get_weights(game_state, action)


class DefenceAgent(BasicAgent):
    def get_features(self, game_state, action):
        super().get_features(game_state, action)

    def get_weights(self, game_state, action):
        super().get_weights(game_state, action)