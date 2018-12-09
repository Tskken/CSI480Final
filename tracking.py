"""Utilities for tracking opponents based on noisy observations.

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
import game
import copy


class InferenceModule:
    """Class for tracking a belief distribution over an opponent's location.

    This is an abstract class.
    """

    def __init__(self, opponent_agent_index, distancer, game_state):
        """Set the opponent agent and distancer for later access."""
        self.index = opponent_agent_index
        self.distancer = distancer

        # store all legal positions
        self.legal_positions = [p for p in
                                game_state.get_walls().as_list(False)]

        # structure used for detecting when opponent has been eaten
        self.possibly_eaten_by = []

        # structures for knowing who is on our team (vs the opponents team)
        if game_state.is_on_red_team(self.index):
            self.our_indices = game_state.get_blue_team_indices()
        else:
            self.our_indices = game_state.get_red_team_indices()

        # opponent starts at their initial position
        self.observe_exact(self.get_initial_position(game_state), game_state)

    def get_position_distribution(self, game_state):
        """Return a distribution over successor positions.

        Specifically this is a distribution over successur position for the
        opponent from the given game_state.

        You must first place the opponent in the game_state, using
        set_opponent_position below.
        """
        opponent_position = game_state.get_agent_position(self.index)
        legal_actions = game_state.get_legal_actions(self.index)

        dist = util.Counter()
        # consider all actions equally probable
        for action in legal_actions:
            successor_position = game.Actions.get_successor(opponent_position,
                                                            action)
            dist[successor_position] = 1.0 / (len(legal_actions))
        return dist

    def set_opponent_position(self, game_state, opponent_position):
        """Put the opponnent at opponent_position in game_state.

        Note that calling set_opponent_position does not change the position of
        the ghost in the game_state object used for tracking the true
        progression of the game.  It will be a deep copy of the true game_state
        object.
        """
        conf = game.Configuration(opponent_position, game.Directions.STOP)
        game_state.data.agent_states[self.index] = game.AgentState(conf, False)
        return game_state

    def observe_state(self, game_state, observer):
        """Collect relevant noisy distance observation and pass it along."""
        true_position = game_state.get_agent_position(self.index)
        if true_position:
            self.observe_exact(true_position, game_state)
        else:
            distances = game_state.get_agent_distances()
            obs = distances[self.index]
            self.obs = obs
            self.observe(obs, game_state, observer)

    def get_initial_position(self, game_state):
        """Get the initial position of this opponent."""
        return game_state.get_initial_agent_position(self.index)

    def get_most_likely_position(self):
        """Return the most likely position for the opponent.

        This is based on the current belief state.
        """
        return self.get_belief_distribution().arg_max()

    ######################################
    # Methods that need to be overridden #
    ######################################

    def observe_exact(self, true_position, game_state):
        """Update beliefs based on an exact observation and game_state."""
        pass

    def observe(self, observation, game_state, observer):
        """Update beliefs based on a distance observation and game_state."""
        pass

    def elapse_time(self, game_state):
        """Update beliefs for a time step elapsing from a game_state."""
        pass

    def get_belief_distribution(self):
        """Return the agent's current belief state.

        This is a distribution over opponent locations conditioned on
        all evidence so far.
        """
        pass


class ExactInference(InferenceModule):
    """This class computes the exact belief function at each time step.

    Uses forward-algorithm updates.
    """

    def observe_exact(self, position, game_state):
        """Update beliefs based on an exact observation and game_state.

        Overrides InferenceModule.observe_exact
        """
        self.beliefs = util.Counter()
        self.beliefs[position] = 1.0

        # check if danger of being eaten
        # simple check to just see whether close to one of our agents, then
        # will detect if our agent was the one eaten later
        # this simplifies edge cases at the center of the board, rather
        # than having a more complex criteria here based on agent types

        self.possibly_eaten_by = []

        for index in self.our_indices:
            if self.distancer.get_distance(game_state.get_agent_position(
                                            index), position) < 2.1:
                # POSSIBLY EATEN BY agent
                self.possibly_eaten_by.append(
                    (index, game_state.get_agent_position(index)))

    def observe(self, observation, game_state, observer):
        """Update beliefs based on a distance observation and game_state.

        The noisy_distance is the estimated Manhattan distance to the
        opponent being tracked.

        Overrides InferenceModule.observe
        """
        if len({key: value for key, value in self.beliefs.items() if value > 0
                }) == 0:
            # THIS SHOULD NOT HAPPEN
            print("***************ALERT: ALL ZEROS********************",
                  self.index)
            # but just in case, we will assume it was because opponent was
            # eaten and we somehow missed that
            self.observe_exact(self.get_initial_position(game_state),
                               game_state)
            return

        if len(self.possibly_eaten_by) > 0:
            eaten = True
            for index, position in self.possibly_eaten_by:
                # if in starting position then we were the one eaten,
                # unless we were already there (in which case, if we can't see
                # the opponent exactly, they were eaten)
                if (((game_state.get_agent_position(index) ==
                      game_state.get_initial_agent_position(index))
                     and (position != game_state.get_agent_position(index)))):
                    # WE WERE EATEN
                    eaten = False
                    break

            if eaten:
                # THE OPPONENT WE ARE TRACKING WAS EATEN, SO THEY RETURNED
                # TO STARTING POSITION
                self.observe_exact(self.get_initial_position(game_state),
                                   game_state)
                return

        # reset this data structure
        self.possibly_eaten_by = []

        noisy_distance = observation
        observer_position = game_state.get_agent_position(observer)

        all_possible = util.Counter()

        for p in [p for p in self.legal_positions if self.beliefs[p] > 0]:
            true_distance = util.manhattan_distance(p, observer_position)
            all_possible[p] = game_state.get_distance_prob(
                    true_distance, noisy_distance) * self.beliefs[p]

        all_possible.normalize()
        self.beliefs = all_possible

    def elapse_time(self, game_state):
        """Update beliefs for a time step elapsing from a game_state.

        Overrides InferenceModule.elapse_time
        """
        # copy the game state, so can place the opponent in different positions
        game_state = copy.deepcopy(game_state)
        all_possible = util.Counter()
        for pos in self.legal_positions:
            new_pos_dist = self.get_position_distribution(
                    self.set_opponent_position(game_state, pos))
            for new_pos, prob in new_pos_dist.items():
                all_possible[new_pos] += prob * self.beliefs[pos]
        all_possible.normalize()
        self.beliefs = all_possible

    def get_belief_distribution(self):
        """Return the agent's current belief state.

        This is a distribution over opponent locations conditioned on
        all evidence so far.

        Overrides InferenceModule.get_belief_distribution
        """
        return self.beliefs
