"""Interfaces for capture agents.

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


from game import Agent
import distance_calculator
from util import nearest_point
import util
import random
import time


class RandomAgent(Agent):
    """A random agent that abides by the rules."""

    def __init__(self, index):
        """Initialize agent with given index."""
        self.index = index

    def get_action(self, state):
        """Return a random legal action."""
        return random.choice(state.get_legal_actions(self.index))


class CaptureAgent(Agent):
    """A base class for capture agents.

    The convenience methods herein handle some of the complications of a
    two-team game.

    Recommended Usage:  Subclass CaptureAgent and override choose_action.
    """

    #############################
    # Methods to store key info #
    #############################

    def __init__(self, index, time_for_computing=.1):
        """Initialize capture agent with several variables you can query.

        self.index = index for this agent
        self.red = true if you're on the red team, false otherwise
        self.agents_on_team = a list of agent objects that make up your team
        self.distancer = distance calculator (contest code provides this)
        self.observation_history = list of GameState objects that correspond
                                   to the sequential order of states that have
                                   occurred so far this game
        self.time_for_computing = an amount of time to give each turn for
                                  computing maze distances
        (part of the provided distance calculator)
        """
        # Agent index for querying state
        self.index = index

        # Whether or not you're on the red team
        self.red = None

        # Agent objects controlling you and your teammates
        self.agents_on_team = None

        # Maze distance calculator
        self.distancer = None

        # A history of observations
        self.observation_history = []

        # Time to spend each turn on computing maze distances
        self.time_for_computing = time_for_computing

        # Access to the graphics
        self.display = None

    def register_initial_state(self, game_state):
        """Handle the initial setup of the agent.

        Populate useful fields (such as what team we're on).

        A distance_calculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.get_distance(p1, p2)
        """
        self.red = game_state.is_on_red_team(self.index)
        self.distancer = distance_calculator.Distancer(game_state.data.layout)

        # comment this out to forgo maze distance computation and
        # use manhattan distances
        self.distancer.get_maze_distances()

        import __main__
        if '_display' in dir(__main__):
            self.display = __main__._display

    def final(self, game_state):
        """Finalize game."""
        self.observation_history = []

    def register_team(self, agents_on_team):
        """Fill the self.agents_on_team field.

        Will be filled with a list of the indices of the agents on your team.
        """
        self.agents_on_team = agents_on_team

    def observation_function(self, game_state):
        """Make an observation about the game_state."""
        return game_state.make_observation(self.index)

    def debug_draw(self, cells, color, clear=False):
        """Draw debug information."""
        if self.display:
            from capture_graphics_display import PacmanGraphics
            if isinstance(self.display, PacmanGraphics):
                if not type(cells) is list:
                    cells = [cells]
                self.display.debug_draw(cells, color, clear)

    def debug_clear(self):
        """Clear out debug information."""
        if self.display:
            from capture_graphics_display import PacmanGraphics
            if isinstance(self.display, PacmanGraphics):
                self.display.clear_debug()

    #################
    # Action Choice #
    #################

    def get_action(self, game_state):
        """Call choose_action on a grid position; continues on half positions.

        If you subclass CaptureAgent, you shouldn't need to override this
        method.  It takes care of appending the current game_state on to your
        observation history (so you have a record of the game states of the
        game) and will call your choose action method if you're in a state
        (rather than halfway through your last move - this occurs because
        Pacman agents move half as quickly as ghost agents).
        """
        self.observation_history.append(game_state)

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        if my_pos != nearest_point(my_pos):
            # We're halfway from one position to the next
            return game_state.get_legal_actions(self.index)[0]
        else:
            return self.choose_action(game_state)

    def choose_action(self, game_state):
        """Override this method to make a good agent.

        It should return a legal action within the time limit (otherwise a
        random legal action will be chosen for you).
        """
        util.raise_not_defined()

    #######################
    # Convenience Methods #
    #######################

    def get_food(self, game_state):
        """Return the food you're meant to eat.

        This is in the form of a matrix
        where m[x][y]=true if there is food you can eat (based on your team)
        in that square.
        """
        if self.red:
            return game_state.get_blue_food()
        else:
            return game_state.get_red_food()

    def get_food_you_are_defending(self, game_state):
        """Return the food you're meant to protect.

        i.e., the food that your opponent is supposed to eat.
        This is in the form of a matrix where m[x][y]=true if
        there is food at (x,y) that your opponent can eat.
        """
        if self.red:
            return game_state.get_red_food()
        else:
            return game_state.get_blue_food()

    def get_capsules(self, game_state):
        """Return the capsule you're meant to eat."""
        if self.red:
            return game_state.get_blue_capsules()
        else:
            return game_state.get_red_capsules()

    def get_capsules_you_are_defending(self, game_state):
        """Return the capsule you're meant to protect."""
        if self.red:
            return game_state.get_red_capsules()
        else:
            return game_state.get_blue_capsules()

    def get_opponents(self, game_state):
        """Return agent indices of your opponents.

        This is the list of the numbers of the agents
        (e.g., red might be "1,3,5")
        """
        if self.red:
            return game_state.get_blue_team_indices()
        else:
            return game_state.get_red_team_indices()

    def get_team(self, game_state):
        """Return agent indices of your team.

        This is the list of the numbers of the agents
        (e.g., red might be the list of 1,3,5)
        """
        if self.red:
            return game_state.get_red_team_indices()
        else:
            return game_state.get_blue_team_indices()

    def get_score(self, game_state):
        """Return how much you are beating the other team by.

        This is in the form of a number that is the difference between your
        score and the opponents score.  This number is negative if you're
        losing.
        """
        if self.red:
            return game_state.get_score()
        else:
            return game_state.get_score() * -1

    def get_maze_distance(self, pos1, pos2):
        """Return the distance between two points.

        These are calculated using the provided distancer object.

        If distancer.get_maze_distances() has been called, then maze distances
        are available.

        Otherwise, this just returns Manhattan distance.
        """
        d = self.distancer.get_distance(pos1, pos2)
        return d

    def get_previous_observation(self):
        """Return GameState object of the last state this agent saw.

        (the observed state of the game last time this agent moved -
        this may not include all of your opponent's agent locations exactly).
        """
        if len(self.observation_history) == 1:
            return None
        else:
            return self.observation_history[-2]

    def get_current_observation(self):
        """Return the GameState object of this agent's current observation.

        (the observed state of the game - this may not include
        all of your opponent's agent locations exactly).
        """
        return self.observation_history[-1]

    def display_distributions_over_positions(self, distributions):
        """Overlays a distribution over positions onto the pacman board.

        This represents an agent's beliefs about the positions of each agent.

        The arg distributions is a tuple or list of util.Counter objects,
        where the i'th Counter has keys that are board positions (x,y) and
        values that encode the probability that agent i is at (x,y).

        If some elements are None, then they will be ignored.
        If a Counter is passed to this function, it will be displayed.

        This is helpful for figuring out if your agent is doing
        inference correctly, and does not affect gameplay.
        """
        dists = []
        for dist in distributions:
            if dist is not None:
                if not isinstance(dist, util.Counter):
                    raise Exception("Wrong type of distribution")
                dists.append(dist)
            else:
                dists.append(util.Counter())

        if ((self.display is not None and
             'update_distributions' in dir(self.display))):
            self.display.update_distributions(dists)
        else:
            self._distributions = dists  # These can be read by pacclient.py


class TimeoutAgent(Agent):
    """A random agent that takes too much time.

    Taking too much time results in penalties and random moves.
    """

    def __init__(self, index):
        """Initialize agent with given index."""
        self.index = index

    def get_action(self, state):
        """Take too much time getting action."""
        time.sleep(2.0)
        return random.choice(state.get_legal_actions(self.index))
