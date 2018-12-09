"""Distance calculator.

This file contains a Distancer object which computes and
caches the shortest path between any two points in the maze.

Example:
distancer = Distancer(game_state.data.layout)
distancer.get_distance( (1,1), (10,10) )

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


class Distancer:
    """Distancer class.

    Computes and caches the shortest path between any two points in the maze.
    """

    def __init__(self, layout, default=10000):
        """Initialize with Distancer(layout).

        Changing default is unnecessary.
        """
        self._distances = None
        self.default = default
        self.dc = DistanceCalculator(layout, self, default)

    def get_maze_distances(self):
        """Compute distances."""
        self.dc.run()

    def get_distance(self, pos1, pos2):
        """Get the distance between two points.

        The only functions you'll need after you create the object.
        """
        if self._distances is None:
            return manhattan_distance(pos1, pos2)
        if is_int(pos1) and is_int(pos2):
            return self.get_distance_on_grid(pos1, pos2)
        pos1_grids = _get_grids2_d(pos1)
        pos2_grids = _get_grids2_d(pos2)
        best_distance = self.default
        for pos1_snap, snap1_distance in pos1_grids:
            for pos2_snap, snap2_distance in pos2_grids:
                grid_distance = self.get_distance_on_grid(pos1_snap, pos2_snap)
                distance = grid_distance + snap1_distance + snap2_distance
                if best_distance > distance:
                    best_distance = distance
        return best_distance

    def get_distance_on_grid(self, pos1, pos2):
        """Get cached distance on the grid."""
        key = (pos1, pos2)
        if key in self._distances:
            return self._distances[key]
        else:
            raise Exception("Positions not in grid: " + str(key))

    def is_ready_for_maze_distance(self):
        """Return if ready to be used for maze distances or not."""
        return self._distances is not None


def manhattan_distance(x, y):
    """Return manhattan distance between x and y."""
    return abs(x[0] - y[0]) + abs(x[1] - y[1])


def is_int(pos):
    """Return if position is made up of integer coordinates."""
    x, y = pos
    return x == int(x) and y == int(y)


def _get_grids2_d(pos):
    grids = []
    for x, x_distance in _get_grids1_d(pos[0]):
        for y, y_distance in _get_grids1_d(pos[1]):
            grids.append(((x, y), x_distance + y_distance))
    return grids


def _get_grids1_d(x):
    int_x = int(x)
    if x == int(x):
        return [(x, 0)]
    return [(int_x, x - int_x), (int_x + 1, int_x + 1 - x)]


##########################################
# MACHINERY FOR COMPUTING MAZE DISTANCES #
##########################################


distance_map = {}


class DistanceCalculator:
    """Distance calculator."""

    def __init__(self, layout, distancer, default=10000):
        """Initialize distance calculator with given layout and distancer."""
        self.layout = layout
        self.distancer = distancer
        self.default = default

    def run(self):
        """Calculate distances."""
        global distance_map

        if self.layout.walls not in distance_map:
            distances = compute_distances(self.layout)
            distance_map[self.layout.walls] = distances
        else:
            distances = distance_map[self.layout.walls]

        self.distancer._distances = distances


def compute_distances(layout):
    """Run UCS to all other positions from each position."""
    distances = {}
    all_nodes = layout.walls.as_list(False)
    for source in all_nodes:
        dist = {}
        closed = {}
        for node in all_nodes:
            dist[node] = float("inf")
        queue = util.PriorityQueue()
        queue.push(source, 0)
        dist[source] = 0
        while not queue.is_empty():
            node = queue.pop()
            if node in closed:
                continue
            closed[node] = True
            node_dist = dist[node]
            adjacent = []
            x, y = node
            if not layout.is_wall((x, y + 1)):
                adjacent.append((x, y + 1))
            if not layout.is_wall((x, y - 1)):
                adjacent.append((x, y - 1))
            if not layout.is_wall((x + 1, y)):
                adjacent.append((x + 1, y))
            if not layout.is_wall((x - 1, y)):
                adjacent.append((x - 1, y))
            for other in adjacent:
                if other not in dist:
                    continue
                old_dist = dist[other]
                new_dist = node_dist + 1
                if new_dist < old_dist:
                    dist[other] = new_dist
                    queue.push(other, new_dist)
        for target in all_nodes:
            distances[(target, source)] = dist[target]
    return distances
