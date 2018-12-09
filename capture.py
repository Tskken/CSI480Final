"""Pacman capture the flag.

(i)  Your interface to the pacman world:
Pacman is a complex environment.  You probably don't want to
read through all of the code we wrote to make the game run
correctly.  This section contains the parts of the code
that you will need to understand in order to complete the
project.  There is also some code in game.py that you should
understand.

(ii)  The hidden secrets of pacman:
This section contains all of the logic code that the pacman
environment uses to decide who can move where, who dies when
things collide, etc.  You shouldn't need to read this section
of code, but you can if you want.

(iii) Framework to start a game:
The final section contains the code for reading the command
you use to set up the game, then starting up a new game, along with
linking in all the external parts (agent functions, graphics).
Check this section out to see all the options available to you.

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

from game import GameStateData
from game import Game
from game import Actions
from util import nearest_point
from util import manhattan_distance
from game import Grid
from game import Configuration
import sys
import util
import time
import random
import imp
import keyboard_agents
from optparse import OptionParser
import traceback

# If you change these, you won't affect the server, so you can't cheat
KILL_POINTS = 0
SONAR_NOISE_RANGE = 13  # Must be odd
SONAR_NOISE_VALUES = [i - (SONAR_NOISE_RANGE - 1) / 2
                      for i in range(SONAR_NOISE_RANGE)]

SIGHT_RANGE = 5  # Manhattan distance
MIN_FOOD = 2
TOTAL_FOOD = 60

# if we have the gameplay element that dumps dots on death
DUMP_FOOD_ON_DEATH = True

SCARED_TIME = 40


def noisy_distance(pos1, pos2):
    """Compute a noisy distance."""
    return int(util.manhattan_distance(pos1, pos2) +
               random.choice(SONAR_NOISE_VALUES))


###################################################
# YOUR INTERFACE TO THE PACMAN WORLD: A GameState #
###################################################


class GameState:
    """A GameState specifies the full game state.

    Full game state includes the food, capsules, agent configurations,
    and score changes.

    GameStates are used by the Game object to capture the actual state of
    the game and can be used by agents to reason about the game.

    Much of the information in a GameState is stored in a GameStateData object.
    We strongly suggest that you access that data via the accessor methods
    below rather than referring to the GameStateData object directly.
    """

    ####################################################
    # Accessor methods: use these to access state data #
    ####################################################

    def get_legal_actions(self, agent_index=0):
        """Return the legal actions for the agent specified."""
        return AgentRules.get_legal_actions(self, agent_index)

    def generate_successor(self, agent_index, action):
        """Return the successor after the specified agent takes the action.

        Returns a GameState object
        """
        # Copy current state
        state = GameState(self)

        # Find appropriate rules for the agent
        AgentRules.apply_action(state, action, agent_index)
        AgentRules.check_death(state, agent_index)
        AgentRules.decrement_timer(state.data.agent_states[agent_index])

        # Book keeping
        state.data._agent_moved = agent_index
        state.data.score += state.data.score_change
        state.data.timeleft = self.data.timeleft - 1
        return state

    def get_agent_state(self, index):
        """Return the state (game.AgentState) of agent with given index."""
        return self.data.agent_states[index]

    def get_agent_position(self, index):
        """Attempt to get agent position.

        Return a location tuple if the agent with the given index is
        observable; if the agent is unobservable, return None.
        """
        agent_state = self.data.agent_states[index]
        ret = agent_state.get_position()
        if ret:
            return tuple(int(x) for x in ret)
        return ret

    def get_num_agents(self):
        """Return number of agents."""
        return len(self.data.agent_states)

    def get_score(self):
        """Return a number corresponding to the current score."""
        return self.data.score

    def get_red_food(self):
        """Return a matrix of food on the red team's side.

        For the matrix m, m[x][y]=true if there is food in (x,y) that belongs
        to red (meaning red is protecting it, blue is trying to eat it).
        """
        return half_grid(self.data.food, red=True)

    def get_blue_food(self):
        """Return a matrix of food on the blue team's side.

        For the matrix m, m[x][y]=true if there is food in (x,y) that belongs
        to blue (meaning blue is protecting it, red is trying to eat it).
        """
        return half_grid(self.data.food, red=False)

    def get_red_capsules(self):
        """Return list of capsule positions (x,y) on the red team's side."""
        return half_list(self.data.capsules, self.data.food, red=True)

    def get_blue_capsules(self):
        """Retun list of capsule positions (x,y) on the blue team's side."""
        return half_list(self.data.capsules, self.data.food, red=False)

    def get_walls(self):
        """Return a matrix of wall positions."""
        return self.data.layout.walls

    def has_food(self, x, y):
        """Return true if the location (x,y) has food.

        This is regardless of whether it's blue team food or red team food.
        """
        return self.data.food[x][y]

    def has_wall(self, x, y):
        """Return true if (x,y) has a wall, false otherwise."""
        return self.data.layout.walls[x][y]

    def is_over(self):
        """Return if game is over."""
        return self.data._win

    def get_red_team_indices(self):
        """Return a list of agent indices for the agents on the red team."""
        return self.red_team[:]

    def get_blue_team_indices(self):
        """Return a list of agent indices for the agents on the blue team."""
        return self.blue_team[:]

    def is_on_red_team(self, agent_index):
        """Return true if the agent is on the red team."""
        return self.teams[agent_index]

    def get_agent_distances(self):
        """Return a noisy distance to each agent."""
        if 'agent_distances' in dir(self):
            return self.agent_distances
        else:
            return None

    def get_distance_prob(self, true_distance, noisy_distance):
        """Return the probability of noisy_distance given true_distance."""
        if noisy_distance - true_distance in SONAR_NOISE_VALUES:
            return 1.0 / SONAR_NOISE_RANGE
        else:
            return 0

    def get_initial_agent_position(self, agent_index):
        """Return the initial position of an agent."""
        return self.data.layout.agent_positions[agent_index][1]

    def get_capsules(self):
        """Return a list of positions (x,y) of the remaining capsules."""
        return self.data.capsules

    #############################################
    #             Helper methods:               #
    # You shouldn't need to call these directly #
    #############################################

    def __init__(self, prev_state=None):
        """Generate a new state by copying information from its predecessor."""
        if prev_state is not None:  # Initial state
            self.data = GameStateData(prev_state.data)
            self.blue_team = prev_state.blue_team
            self.red_team = prev_state.red_team
            self.data.timeleft = prev_state.data.timeleft

            self.teams = prev_state.teams
            self.agent_distances = prev_state.agent_distances
        else:
            self.data = GameStateData()
            self.agent_distances = []

    def deep_copy(self):
        """Return a deep copy of this state."""
        state = GameState(self)
        state.data = self.data.deep_copy()
        state.data.timeleft = self.data.timeleft

        state.blue_team = self.blue_team[:]
        state.red_team = self.red_team[:]
        state.teams = self.teams[:]
        state.agent_distances = self.agent_distances[:]
        return state

    def make_observation(self, index):
        """Make observation for agent with given index.

        Return a GameState instance.
        """
        state = self.deep_copy()

        # Adds the sonar signal
        pos = state.get_agent_position(index)
        n = state.get_num_agents()
        distances = [noisy_distance(pos, state.get_agent_position(i))
                     for i in range(n)]
        state.agent_distances = distances

        # Remove states of distant opponents
        if index in self.blue_team:
            team = self.blue_team
            other_team = self.red_team
        else:
            other_team = self.blue_team
            team = self.red_team

        for enemy in other_team:
            seen = False
            enemy_pos = state.get_agent_position(enemy)
            for teammate in team:
                if ((util.manhattan_distance(
                        enemy_pos, state.get_agent_position(teammate))
                     <= SIGHT_RANGE)):
                    seen = True
            if not seen:
                state.data.agent_states[enemy].configuration = None
        return state

    def __eq__(self, other):
        """Allow two states to be compared."""
        if other is None:
            return False
        return self.data == other.data

    def __hash__(self):
        """Allow states to be keys of dictionaries."""
        return int(hash(self.data))

    def __str__(self):
        """Allow conversion to string."""
        return str(self.data)

    def initialize(self, layout, num_agents):
        """Create an initial game state from a layout array (see layout.py)."""
        self.data.initialize(layout, num_agents)
        positions = [a.configuration for a in self.data.agent_states]
        self.blue_team = [i for i, p in enumerate(positions)
                          if not self.is_red(p)]
        self.red_team = [i for i, p in enumerate(positions) if self.is_red(p)]
        self.teams = [self.is_red(p) for p in positions]
        # This is usually 60 (always 60 with random maps)
        # However, if layout map is specified otherwise, it could be less
        global TOTAL_FOOD
        TOTAL_FOOD = layout.total_food

    def is_red(self, config_or_pos):
        """Determine if position in on the red side of the board."""
        width = self.data.layout.width
        if isinstance(config_or_pos, tuple):
            return config_or_pos[0] < width / 2
        else:
            return config_or_pos.position[0] < width / 2


def half_grid(grid, red):
    """Return a half grid for red or blue.

    A half grid is the same size as the given full grid, but only values on one
    horizontal half are populated.
    """
    halfway = grid.width / 2
    halfgrid = Grid(grid.width, grid.height, False)
    if red:
        xrange = list(range(int(halfway)))
    else:
        xrange = list(range(int(halfway), grid.width))

    for y in range(grid.height):
        for x in xrange:
            if grid[x][y]:
                halfgrid[x][y] = True

    return halfgrid


def half_list(l, grid, red):
    """Return list of positions that are in l and on the specified side."""
    halfway = grid.width / 2
    new_list = []
    for x, y in l:
        if red and x <= halfway:
            new_list.append((x, y))
        elif not red and x > halfway:
            new_list.append((x, y))
    return new_list


############################################################################
#                     THE HIDDEN SECRETS OF PACMAN                         #
#                                                                          #
# You shouldn't need to look through the code in this section of the file. #
############################################################################

COLLISION_TOLERANCE = 0.7  # How close ghosts must be to Pacman to kill


class CaptureRules:
    """These game rules manage the control flow of a game.

    They decide when and how the game starts and ends.
    """

    def __init__(self, quiet=False):
        """Create the CaptureRules, specifying whether to be quiet.

        If quiet is true, will not print output information.
        """
        self.quiet = quiet

    def new_game(self, layout, agents, display, length, mute_agents,
                 catch_exceptions):
        """Create a new game.Game from the provided arguments."""
        init_state = GameState()
        init_state.initialize(layout, len(agents))
        starter = random.randint(0, 1)
        print(('%s team starts' % ['Red', 'Blue'][starter]))
        game = Game(agents, display, self, starting_index=starter,
                    mute_agents=mute_agents, catch_exceptions=catch_exceptions)
        game.state = init_state
        game.length = length
        game.state.data.timeleft = length
        if 'draw_center_line' in dir(display):
            display.draw_center_line()
        self._init_blue_food = init_state.get_blue_food().count()
        self._init_red_food = init_state.get_red_food().count()
        return game

    def process(self, state, game):
        """Check to see whether it is time to end the game."""
        if 'move_history' in dir(game):
            if len(game.move_history) == game.length:
                state.data._win = True

        if state.is_over():
            game.game_over = True
            if not game.rules.quiet:
                red_count = 0
                blue_count = 0
                food_to_win = (TOTAL_FOOD / 2) - MIN_FOOD
                for index in range(state.get_num_agents()):
                    agent_state = state.data.agent_states[index]
                    if index in state.get_red_team_indices():
                        red_count += agent_state.num_returned
                    else:
                        blue_count += agent_state.num_returned

                if blue_count >= food_to_win:
                    print('The Blue team has returned at least '
                          '%d of the opponents\' dots.' % food_to_win)
                elif red_count >= food_to_win:
                    print('The Red team has returned at least '
                          '%d of the opponents\' dots.' % food_to_win)
                else:
                    print('Time is up.')
                    if state.data.score == 0:
                        print('Tie game!')
                    else:
                        winner = 'Red'
                        if state.data.score < 0:
                            winner = 'Blue'
                        print('The %s team wins by %d points.' %
                              (winner, abs(state.data.score)))

    def get_progress(self, game):
        """Return the most likely progress indicator, clamped to [0, 1]."""
        blue = 1.0 - (game.state.get_blue_food().count() /
                      float(self._init_blue_food))
        red = 1.0 - (game.state.get_red_food().count() /
                     float(self._init_red_food))
        moves = len(self.move_history) / float(game.length)

        return min(max(0.75 * max(red, blue) + 0.25 * moves, 0.0), 1.0)

    def agent_crash(self, game, agent_index):
        """Handle agent crashing."""
        if agent_index % 2 == 0:
            print("Red agent crashed")
            game.state.data.score = -1
        else:
            print("Blue agent crashed")
            game.state.data.score = 1

    def get_max_total_time(self, agent_index):
        """Return max amount of total time.

        Move limits should prevent this from ever happening.
        """
        return 900

    def get_max_startup_time(self, agent_index):
        """Return max allowed time for register_initial_state.

        Here defined to be 15 seconds.
        """
        return 15

    def get_move_warning_time(self, agent_index):
        """Return amount of time per move, which if exceeded incurs a warning.

        Here defined to be one second.
        """
        return 1

    def get_move_timeout(self, agent_index):
        """Return max amount of time per move before instant forfeit.

        Here defined to be three seconds.
        """
        return 3  # Three seconds results in instant forfeit

    def get_max_time_warnings(self, agent_index):
        """Return the max number time warnings before forfeit.

        Here defined to be 2 (so third warning results in forfeit.
        """
        return 2  # Third violation loses the game


class AgentRules:
    """These functions govern how each agent interacts with her environment."""

    @staticmethod
    def get_legal_actions(state, agent_index):
        """Return a list of legal actions (both possible & allowed)."""
        agent_state = state.get_agent_state(agent_index)
        conf = agent_state.configuration
        possible_actions = Actions.get_possible_actions(
                                conf, state.data.layout.walls)
        return AgentRules.filter_for_allowed_actions(agent_state,
                                                     possible_actions)

    @staticmethod
    def filter_for_allowed_actions(agent_state, possible_actions):
        """Filter possible actions for allowed actions."""
        return possible_actions

    @staticmethod
    def apply_action(state, action, agent_index):
        """Edit the state to reflect the results of the action."""
        legal = AgentRules.get_legal_actions(state, agent_index)
        if action not in legal:
            raise Exception("Illegal action " + str(action))

        # Update Configuration
        agent_state = state.data.agent_states[agent_index]
        speed = 1.0
        # if agent_state.is_pacman: speed = 0.5
        vector = Actions.direction_to_vector(action, speed)
        old_config = agent_state.configuration
        agent_state.configuration = old_config.generate_successor(vector)

        # Eat
        next = agent_state.configuration.get_position()
        nearest = nearest_point(next)

        if next == nearest:
            is_red = state.is_on_red_team(agent_index)
            # Change agent type
            agent_state.is_pacman =\
                [is_red, state.is_red(agent_state.configuration)
                 ].count(True) == 1
            # if he's no longer pacman, he's on his own side,
            # so reset the num carrying timer
            # agent_state.num_carrying *= int(agent_state.is_pacman)
            if agent_state.num_carrying > 0 and not agent_state.is_pacman:
                score = (agent_state.num_carrying if is_red
                         else -1 * agent_state.num_carrying)
                state.data.score_change += score

                agent_state.num_returned += agent_state.num_carrying
                agent_state.num_carrying = 0

                red_count = 0
                blue_count = 0
                for index in range(state.get_num_agents()):
                    agent_state = state.data.agent_states[index]
                    if index in state.get_red_team_indices():
                        red_count += agent_state.num_returned
                    else:
                        blue_count += agent_state.num_returned
                if (((red_count >= (TOTAL_FOOD / 2) - MIN_FOOD) or
                     (blue_count >= (TOTAL_FOOD / 2) - MIN_FOOD))):
                    state.data._win = True

        if agent_state.is_pacman and manhattan_distance(nearest, next) <= 0.9:
            AgentRules.consume(nearest, state,
                               state.is_on_red_team(agent_index))

    @staticmethod
    def consume(position, state, is_red):
        """Handle consuming food/capsule (called from apply_action)."""
        x, y = position
        # Eat food
        if state.data.food[x][y]:

            # blue case is the default
            team_indices_func = state.get_blue_team_indices
            # score = -1
            if is_red:
                # switch if its red
                # score = 1
                team_indices_func = state.get_red_team_indices

            # go increase the variable for the pacman who ate this
            agents = [state.data.agent_states[agent_index]
                      for agent_index in team_indices_func()]
            for agent in agents:
                if agent.get_position() == position:
                    agent.num_carrying += 1
                    break  # the above should only be true for one agent...

            # do all the score and food grid maintainenace
            # state.data.score_change += score
            state.data.food = state.data.food.copy()
            state.data.food[x][y] = False
            state.data._food_eaten = position
            # if ((is_red and state.get_blue_food().count() == MIN_FOOD) or
            #     (not is_red and state.get_red_food().count() == MIN_FOOD)):
            #    state.data._win = True

        # Eat capsule
        if is_red:
            my_capsules = state.get_blue_capsules()
        else:
            my_capsules = state.get_red_capsules()
        if(position in my_capsules):
            state.data.capsules.remove(position)
            state.data._capsule_eaten = position

            # Reset all ghosts' scared timers
            if is_red:
                other_team = state.get_blue_team_indices()
            else:
                other_team = state.get_red_team_indices()
            for index in other_team:
                state.data.agent_states[index].scared_timer = SCARED_TIME

    @staticmethod
    def decrement_timer(state):
        """Decrement ghost's scared timer."""
        timer = state.scared_timer
        if timer == 1:
            state.configuration.position =\
                nearest_point(state.configuration.position)
        state.scared_timer = max(0, timer - 1)

    @staticmethod
    def dump_food_from_death(state, agent_state, agent_index):
        """Dump a pacman's food when it is eaten."""
        if not (DUMP_FOOD_ON_DEATH):
            # this feature is not turned on
            return

        if not agent_state.is_pacman:
            raise Exception('something is seriously wrong, '
                            'this agent isnt a pacman!')

        # ok so agent_state is this:
        if (agent_state.num_carrying == 0):
            return

        # first, score changes!
        # we HACK pack that ugly bug by just determining if its
        # red based on the first position to die...
        dummy_config = Configuration(agent_state.get_position(), 'North')
        is_red = state.is_red(dummy_config)

        # the score increases if red eats dots, so if we are refunding points,
        # the direction should be -1 if the red agent died, which means he dies
        # on the blue side
        # score_direction = (-1) ** (int(is_red) + 1)
        # state.data.score_change += score_direction * agent_state.num_carrying

        def on_right_side(state, x, y):
            dummy_config = Configuration((x, y), 'North')
            return state.is_red(dummy_config) == is_red

        # we have food to dump
        # -- expand out in BFS. Check:
        #   - that it's within the limits
        #   - that it's not a wall
        #   - that no other agents are there
        #   - that no power pellets are there
        #   - that it's on the right side of the grid
        def all_good(state, x, y):
            width, height = state.data.layout.width, state.data.layout.height
            food, walls = state.data.food, state.data.layout.walls

            # bounds check
            if x >= width or y >= height or x <= 0 or y <= 0:
                return False

            if walls[x][y]:
                return False
            if food[x][y]:
                return False

            # dots need to be on the side where this agent will be a pacman :P
            if not on_right_side(state, x, y):
                return False

            if (x, y) in state.data.capsules:
                return False

            # loop through agents
            agent_poses = [state.get_agent_position(i)
                           for i in range(state.get_num_agents())]
            if (x, y) in agent_poses:
                return False

            return True

        num_to_dump = agent_state.num_carrying
        state.data.food = state.data.food.copy()
        food_added = []

        def gen_successors(x, y):
            dxs = [-1, 0, 1]
            dys = [-1, 0, 1]
            return [(x + dx, y + dy) for dx in dxs for dy in dys]

        # BFS graph search
        position_queue = [agent_state.get_position()]
        seen = set()
        while num_to_dump > 0:
            if not len(position_queue):
                raise Exception('Exhausted BFS! uh oh')
            # pop one off, graph check
            popped = position_queue.pop(0)
            if popped in seen:
                continue
            seen.add(popped)

            x, y = popped[0], popped[1]
            x = int(x)
            y = int(y)
            if (all_good(state, x, y)):
                state.data.food[x][y] = True
                food_added.append((x, y))
                num_to_dump -= 1

            # generate successors
            position_queue = position_queue + gen_successors(x, y)

        state.data._food_added = food_added
        # now our agent_state is no longer carrying food
        agent_state.num_carrying = 0

    @staticmethod
    def check_death(state, agent_index):
        """Resolve multi-agent effects after specified agent moved."""
        agent_state = state.data.agent_states[agent_index]
        if state.is_on_red_team(agent_index):
            other_team = state.get_blue_team_indices()
        else:
            other_team = state.get_red_team_indices()
        if agent_state.is_pacman:
            for index in other_team:
                other_agent_state = state.data.agent_states[index]
                if other_agent_state.is_pacman:
                    continue
                ghost_position = other_agent_state.get_position()
                if ghost_position is None:
                    continue
                if manhattan_distance(ghost_position,
                                      agent_state.get_position()
                                      ) <= COLLISION_TOLERANCE:
                    # award points to the other team for killing Pacmen
                    if other_agent_state.scared_timer <= 0:
                        AgentRules.dump_food_from_death(state,
                                                        agent_state,
                                                        agent_index)

                        score = KILL_POINTS
                        if state.is_on_red_team(agent_index):
                            score = -score
                        state.data.score_change += score
                        agent_state.is_pacman = False
                        agent_state.configuration = agent_state.start
                        agent_state.scared_timer = 0
                    else:
                        score = KILL_POINTS
                        if state.is_on_red_team(agent_index):
                            score = -score
                        state.data.score_change += score
                        other_agent_state.is_pacman = False
                        other_agent_state.configuration =\
                            other_agent_state.start
                        other_agent_state.scared_timer = 0
        else:  # Agent is a ghost
            for index in other_team:
                other_agent_state = state.data.agent_states[index]
                if not other_agent_state.is_pacman:
                    continue
                pac_pos = other_agent_state.get_position()
                if pac_pos is None:
                    continue
                if ((manhattan_distance(pac_pos, agent_state.get_position())
                     <= COLLISION_TOLERANCE)):
                    # award points to the other team for killing Pacmen
                    if agent_state.scared_timer <= 0:
                        AgentRules.dump_food_from_death(state,
                                                        other_agent_state,
                                                        agent_index)

                        score = KILL_POINTS
                        if not state.is_on_red_team(agent_index):
                            score = -score
                        state.data.score_change += score
                        other_agent_state.is_pacman = False
                        other_agent_state.configuration =\
                            other_agent_state.start
                        other_agent_state.scared_timer = 0
                    else:
                        score = KILL_POINTS
                        if state.is_on_red_team(agent_index):
                            score = -score
                        state.data.score_change += score
                        agent_state.is_pacman = False
                        agent_state.configuration = agent_state.start
                        agent_state.scared_timer = 0

    @staticmethod
    def place_ghost(state, ghost_state):
        """Place ghost in starting configuration."""
        ghost_state.configuration = ghost_state.start


#############################
# FRAMEWORK TO START A GAME #
#############################


def default(string):
    """Format help string with default."""
    return string + ' [Default: %default]'


def parse_agent_args(string):
    """Parse extra arguments for agents."""
    if string is None or string == '':
        return {}
    pieces = string.split(',')
    opts = {}
    for p in pieces:
        if '=' in p:
            key, val = p.split('=')
        else:
            key, val = p, 1
        opts[key] = val
    return opts


def read_command(argv):
    """Process the command used to run pacman from the command line."""
    usage_str = """
USAGE:      python pacman.py <options>
EXAMPLES:   (1) python capture.py
                - starts a game with two baseline agents
            (2) python capture.py --keys0
                - starts a two-player interactive game where the arrow keys
                  control agent 0 and all other agents are baseline agents
            (3) python capture.py -r baseline_team -b my_team
                - starts a fully automated game where the red team is a
                  baseline team and blue team is my_team
"""
    parser = OptionParser(usage_str)

    parser.add_option('-r', '--red', help=default('Red team'),
                      default='baseline_team')
    parser.add_option('-b', '--blue', help=default('Blue team'),
                      default='baseline_team')
    parser.add_option('--red-name', help=default('Red team name'),
                      default='Red')
    parser.add_option('--blue-name', help=default('Blue team name'),
                      default='Blue')
    parser.add_option('--red_opts',
                      help=default('Options for red team (e.g. first=keys)'),
                      default='')
    parser.add_option('--blue_opts',
                      help=default('Options for blue team (e.g. first=keys)'),
                      default='')
    parser.add_option('--keys0',
                      help='Make agent 0 (first red player) a keyboard agent',
                      action='store_true', default=False)
    parser.add_option('--keys1',
                      help='Make agent 1 (second red player) a keyboard agent',
                      action='store_true', default=False)
    parser.add_option('--keys2',
                      help='Make agent 2 (first blue player) a keyboard agent',
                      action='store_true', default=False)
    parser.add_option('--keys3',
                      help='Make agent 3 (second blue player) '
                           'a keyboard agent',
                      action='store_true', default=False)
    parser.add_option('-l', '--layout', dest='layout',
                      help=default('the LAYOUT_FILE from which to load the '
                                   'map layout; use RANDOM for a random maze; '
                                   'use RANDOM<seed> to use a specified '
                                   'random seed, e.g., RANDOM23'),
                      metavar='LAYOUT_FILE', default='default_capture')
    parser.add_option('-t', '--textgraphics', action='store_true',
                      dest='textgraphics', help='Display output as text only',
                      default=False)

    parser.add_option('-q', '--quiet', action='store_true',
                      help='Display minimal output and no graphics',
                      default=False)

    parser.add_option('-Q', '--super-quiet', action='store_true',
                      dest="super_quiet",
                      help='Same as -q but agent output is also suppressed',
                      default=False)

    parser.add_option('-z', '--zoom', type='float', dest='zoom',
                      help=default('Zoom in the graphics'), default=1)
    parser.add_option('-i', '--time', type='int', dest='time',
                      help=default('TIME limit of a game in moves'),
                      default=1200, metavar='TIME')
    parser.add_option('-n', '--num_games', type='int',
                      help=default('Number of games to play'), default=1)
    parser.add_option('-f', '--fix_random_seed', action='store_true',
                      help='Fixes the random seed to always '
                           'play the same game', default=False)
    parser.add_option('--record', action='store_true',
                      help='Writes game histories to a file '
                      '(named by the time they were played)', default=False)
    parser.add_option('--replay', default=None,
                      help='Replays a recorded game file.')
    parser.add_option('--replay_delay_time', default=0, type='int',
                      help='Delay time between each action while replaying '
                           '(in ms)')
    parser.add_option('-x', '--num_training', dest='num_training', type='int',
                      help=default('How many episodes are training '
                                   '(suppresses output)'), default=0)
    parser.add_option('-c', '--catch_exceptions', action='store_true',
                      default=False,
                      help='Catch exceptions and enforce time limits')

    parser.add_option('--record_to', default=None)

    options, otherjunk = parser.parse_args(argv)
    assert len(otherjunk) == 0, "Unrecognized options: " + str(otherjunk)
    args = dict()

    # Choose a display format
    # if options.pygame:
    #     import pygame_display
    #     args['display'] = pygame_display.PacmanGraphics()
    if options.textgraphics:
        import text_display
        args['display'] = text_display.PacmanGraphics()
    elif options.quiet:
        import text_display
        args['display'] = text_display.NullGraphics()
    elif options.super_quiet:
        import text_display
        args['display'] = text_display.NullGraphics()
        args['mute_agents'] = True
    else:
        import capture_graphics_display
        # Hack for agents writing to the display
        capture_graphics_display.FRAME_TIME = 0
        args['display'] = capture_graphics_display.PacmanGraphics(
                options.red, options.blue, options.zoom, 0, capture=True)
        import __main__
        __main__.__dict__['_display'] = args['display']

    args['red_team_name'] = options.red_name
    args['blue_team_name'] = options.blue_name

    if options.fix_random_seed:
        random.seed('CSI480')

    # Special case: recorded games don't use the run_games method or
    # args structure
    if options.replay is not None:
        print('Replaying recorded game %s.' % options.replay)
        import pickle
        recorded = pickle.load(open(options.replay, "rb"))
        recorded['display'] = args['display']
        recorded['delay_time'] = options.replay_delay_time
        replay_game(**recorded)
        sys.exit(0)

    # Choose a pacman agent
    red_args = parse_agent_args(options.red_opts)
    blue_args = parse_agent_args(options.blue_opts)
    if options.num_training > 0:
        red_args['num_training'] = options.num_training
        blue_args['num_training'] = options.num_training
    nokeyboard = (options.textgraphics or options.quiet or
                  options.num_training > 0)
    print('\nRed team %s with %s:' % (options.red, red_args))
    red_agents = load_agents(True, options.red, nokeyboard, red_args)
    print('\nBlue team %s with %s:' % (options.blue, blue_args))
    blue_agents = load_agents(False, options.blue, nokeyboard, blue_args)
    # list of agents
    args['agents'] = sum([list(el) for el in zip(red_agents, blue_agents)], [])

    num_keyboard_agents = 0
    for index, val in enumerate([options.keys0, options.keys1, options.keys2,
                                 options.keys3]):
        if not val:
            continue
        if num_keyboard_agents == 0:
            agent = keyboard_agents.KeyboardAgent(index)
        elif num_keyboard_agents == 1:
            agent = keyboard_agents.KeyboardAgent2(index)
        else:
            raise Exception('Max of two keyboard agents supported')
        num_keyboard_agents += 1
        args['agents'][index] = agent

    # Choose a layout
    from layout import Layout, get_layout
    layouts = []
    for i in range(options.num_games):
        if options.layout == 'RANDOM':
            layout = Layout(random_layout().split('\n'))
        elif options.layout.startswith('RANDOM'):
            layout = Layout(random_layout(int(options.layout[6:])).split('\n'))
        elif options.layout.lower().find('capture') == -1:
            raise Exception('You must use a capture layout with capture.py')
        else:
            layout = get_layout(options.layout)
        if layout is None:
            raise Exception("The layout " + options.layout +
                            " cannot be found")

        layouts.append(layout)

    args['layouts'] = layouts
    args['length'] = options.time
    args['num_games'] = options.num_games
    args['num_training'] = options.num_training
    args['record'] = options.record
    args['catch_exceptions'] = options.catch_exceptions

    args['record_to'] = options.record_to

    return args


def random_layout(seed=None):
    """Generate random layout."""
    if not seed:
        seed = random.randint(0, 99999999)
    # layout = 'layouts/random%08dCapture.lay' % seed
    # print 'Generating random layout in %s' % layout
    import maze_generator
    return maze_generator.generate_maze(seed)


def load_agents(is_red, factory, textgraphics, cmd_line_args):
    """Call agent factories and returns lists of agents."""
    try:
        if not factory.endswith(".py"):
            factory += ".py"

        module = imp.load_source('player' + str(int(is_red)), factory)
    except (NameError, ImportError):
        print('Error: The team "' + factory + '" could not be loaded! ',
              file=sys.stderr)
        traceback.print_exc()
        return [None for i in range(2)]

    args = dict()
    args.update(cmd_line_args)  # Add command line args with priority

    print("Loading Team:", factory)
    print("Arguments:", args)

    # if textgraphics and factory_class_name.startswith('Keyboard'):
    #   raise Exception('Using the keyboard requires graphics '
    #                   '(no text display, quiet or training games)')

    try:
        create_team_func = getattr(module, 'create_team')
    except AttributeError:
        print('Error: The team "' + factory + '" could not be loaded! ',
              file=sys.stderr)
        traceback.print_exc()
        return [None for i in range(2)]

    index_addend = 0
    if not is_red:
        index_addend = 1
    indices = [2 * i + index_addend for i in range(2)]
    return create_team_func(indices[0], indices[1], is_red, **args)


def replay_game(layout, agents, actions, display, length, red_team_name,
                blue_team_name, delay_time=0):
    """Replay the specified game."""
    rules = CaptureRules()
    game = rules.new_game(layout, agents, display, length, False, False)
    state = game.state
    display.red_team = red_team_name
    display.blue_team = blue_team_name
    display.initialize(state.data)

    for action in actions:
        # Execute the action
        state = state.generate_successor(*action)
        # Change the display
        display.update(state.data)
        # Allow for game specific conditions (winning, losing, etc.)
        rules.process(state, game)
        if delay_time > 0:
            time.sleep(delay_time/1000.0)

    display.finish()


def run_games(layouts, agents, display, length, num_games, record,
              num_training, red_team_name, blue_team_name, mute_agents=False,
              catch_exceptions=False, record_to=None):
    """Run 1 or more games."""
    rules = CaptureRules()
    games = []

    if num_training > 0:
        print('Playing %d training games' % num_training)

    for i in range(num_games):
        be_quiet = i < num_training
        layout = layouts[i]
        if be_quiet:
            # Suppress output and graphics
            import text_display
            game_display = text_display.NullGraphics()
            rules.quiet = True
        else:
            game_display = display
            rules.quiet = False
        g = rules.new_game(layout, agents, game_display, length, mute_agents,
                           catch_exceptions)
        g.run()
        if not be_quiet:
            games.append(g)

        g.record = None
        print(red_team_name, blue_team_name)
        if record:
            import pickle
            import game
            # fname = ('recorded-game-%d' % (i + 1)) +
            #          '-'.join([str(t) for t in time.localtime()[1:6]])
            # f = file(fname, 'w')
            components = {'layout': layout,
                          'agents': [game.Agent() for a in agents],
                          'actions': g.move_history, 'length': length,
                          'red_team_name': red_team_name,
                          'blue_team_name': blue_team_name}
            # f.close()
            print("recorded")
            g.record = pickle.dumps(components)

            if record_to is None:
                record_to = 'replay'

            with open(record_to + "_" + str(i), 'wb') as f:
                f.write(g.record)

    if num_games > 1:
        scores = [game.state.data.score for game in games]
        red_win_rate = [s > 0 for s in scores].count(True) / float(len(scores))
        blue_win_rate = ([s < 0 for s in scores].count(True) /
                         float(len(scores)))
        print('Average Score:', sum(scores) / float(len(scores)))
        print('Scores:       ', ', '.join([str(score) for score in scores]))
        print('Red Win Rate:  %d/%d (%.2f)' %
              ([s > 0 for s in scores].count(True), len(scores), red_win_rate))
        print('Blue Win Rate: %d/%d (%.2f)' %
              ([s < 0 for s in scores].count(True),
               len(scores), blue_win_rate))
        print('Record:       ', ', '.join([('Blue', 'Tie', 'Red'
                                            )[max(0, min(2, 1 + s))]
                                           for s in scores]))
    return games


def save_score(game):
    """Write score to file."""
    with open('score', 'w') as f:
        print(game.state.data.score, file=f)


def main():
    """Run capture.py from the command line.

    > python capture.py

    See the usage string for more details.

    > python capture.py --help
    """
    options = read_command(sys.argv[1:])  # Get game components based on input
    games = run_games(**options)

    save_score(games[0])
    # import cProfile
    # cProfile.run('run_games( **options )', 'profile')


if __name__ == '__main__':
    main()
