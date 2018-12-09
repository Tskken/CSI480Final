"""Keyboard interfaces to control Pacman.

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
from game import Directions
import random


class KeyboardAgent(Agent):
    """An agent controlled by the keyboard.

    Controlled by a,d,w,s,q -OR- by arrow keys
    """

    WEST_KEY = 'a'
    EAST_KEY = 'd'
    NORTH_KEY = 'w'
    SOUTH_KEY = 's'
    STOP_KEY = 'q'

    def __init__(self, index=0):
        """Initialize keyboard agent."""
        super().__init__(index)
        self.last_move = Directions.STOP
        self.keys = []

    def get_action(self, state):
        """Get action from the keyboard."""
        from graphics_utils import keys_waiting
        from graphics_utils import keys_pressed
        keys = keys_waiting() + keys_pressed()
        if keys != []:
            self.keys = keys

        legal = state.get_legal_actions(self.index)
        move = self.get_move(legal)

        if move == Directions.STOP:
            # Try to move in the same direction as before
            if self.last_move in legal:
                move = self.last_move

        if (self.STOP_KEY in self.keys) and Directions.STOP in legal:
            move = Directions.STOP

        if move not in legal:
            move = random.choice(legal)

        self.last_move = move
        return move

    def get_move(self, legal):
        """Get move from keys and legal actions."""
        move = Directions.STOP
        if ((self.WEST_KEY in self.keys or 'Left' in self.keys) and
                Directions.WEST in legal):
            move = Directions.WEST
        if ((self.EAST_KEY in self.keys or 'Right' in self.keys) and
                Directions.EAST in legal):
            move = Directions.EAST
        if ((self.NORTH_KEY in self.keys or 'Up' in self.keys) and
                Directions.NORTH in legal):
            move = Directions.NORTH
        if ((self.SOUTH_KEY in self.keys or 'Down' in self.keys) and
                Directions.SOUTH in legal):
            move = Directions.SOUTH
        return move


class KeyboardAgent2(KeyboardAgent):
    """A second agent controlled by the keyboard.

    Controlled by j,l,i,k,u
    """

    WEST_KEY = 'j'
    EAST_KEY = "l"
    NORTH_KEY = 'i'
    SOUTH_KEY = 'k'
    STOP_KEY = 'u'

    def get_move(self, legal):
        """Get move from keys and legal actions.

        Overrides KeyboardAgent.get_move
        """
        move = Directions.STOP
        if (self.WEST_KEY in self.keys) and Directions.WEST in legal:
            move = Directions.WEST
        if (self.EAST_KEY in self.keys) and Directions.EAST in legal:
            move = Directions.EAST
        if (self.NORTH_KEY in self.keys) and Directions.NORTH in legal:
            move = Directions.NORTH
        if (self.SOUTH_KEY in self.keys) and Directions.SOUTH in legal:
            move = Directions.SOUTH
        return move
