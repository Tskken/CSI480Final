"""Graphical display for Pacman.

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

Most code by Dan Klein and John Denero written or rewritten for cs188,
UC Berkeley.  Some code from a Pacman implementation by LiveWires,
and used / modified with permission.
"""

import graphics_utils as gu
import math
from game import Directions
import os
import time


DEFAULT_GRID_SIZE = 30.0
INFO_PANE_HEIGHT = 35
BACKGROUND_COLOR = gu.format_color(0, 0, 0)
WALL_COLOR = gu.format_color(0.0 / 255.0, 51.0 / 255.0, 255.0 / 255.0)
INFO_PANE_COLOR = gu.format_color(.4, .4, 0)
SCORE_COLOR = gu.format_color(.9, .9, .9)
PACMAN_OUTLINE_WIDTH = 2
PACMAN_CAPTURE_OUTLINE_WIDTH = 4

GHOST_COLORS = []
GHOST_COLORS.append(gu.format_color(.9, 0, 0))  # Red
GHOST_COLORS.append(gu.format_color(0, .3, .9))  # Blue
GHOST_COLORS.append(gu.format_color(.98, .41, .07))  # Orange
GHOST_COLORS.append(gu.format_color(.1, .75, .7))  # Green
GHOST_COLORS.append(gu.format_color(1.0, 0.6, 0.0))  # Yellow
GHOST_COLORS.append(gu.format_color(.4, 0.13, 0.91))  # Purple

TEAM_COLORS = GHOST_COLORS[:2]

GHOST_SHAPE = [
    (0, 0.3),
    (0.25, 0.75),
    (0.5, 0.3),
    (0.75, 0.75),
    (0.75, -0.5),
    (0.5, -0.75),
    (-0.5, -0.75),
    (-0.75, -0.5),
    (-0.75, 0.75),
    (-0.5, 0.3),
    (-0.25, 0.75)
  ]
GHOST_SIZE = 0.65
SCARED_COLOR = gu.format_color(1, 1, 1)

GHOST_VEC_COLORS = list(map(gu.color_to_vector, GHOST_COLORS))

PACMAN_COLOR = gu.format_color(255.0 / 255.0, 255.0 / 255.0, 61.0 / 255)
PACMAN_SCALE = 0.5

# Food
FOOD_COLOR = gu.format_color(1, 1, 1)
FOOD_SIZE = 0.1

# Laser
LASER_COLOR = gu.format_color(1, 0, 0)
LASER_SIZE = 0.02

# Capsule graphics
CAPSULE_COLOR = gu.format_color(1, 1, 1)
CAPSULE_SIZE = 0.25

# Drawing walls
WALL_RADIUS = 0.15

WHITE = gu.format_color(1.0, 1.0, 1.0)
BLACK = gu.format_color(0.0, 0.0, 0.0)


class InfoPane:
    """Information pane for displaying score."""

    def __init__(self, layout, grid_size):
        """Create InfoPane based on given layout and grid_size."""
        self.grid_size = grid_size
        self.width = (layout.width) * grid_size
        self.base = (layout.height + 1) * grid_size
        self.height = INFO_PANE_HEIGHT
        self.font_size = 24
        self.text_color = PACMAN_COLOR
        self.draw_pane()

    def to_screen(self, pos, y=None):
        """Translate a point relative from the bottom left of the info pane."""
        if y is None:
            x, y = pos
        else:
            x = pos

        x = self.grid_size + x  # Margin
        y = self.base + y
        return x, y

    def draw_pane(self):
        """Draw the pane to screen with score 0."""
        self.score_text = gu.text(self.to_screen(0, 0), self.text_color,
                                  "SCORE:    0", "Times", self.font_size,
                                  "bold")

    def initialize_ghost_distances(self, distances):
        """Initialize display for showing ghost distances."""
        self.ghost_distance_text = []

        size = 20
        if self.width < 240:
            size = 12
        if self.width < 160:
            size = 10

        for i, d in enumerate(distances):
            t = gu.text(self.to_screen(self.width / 2 + self.width / 8 * i, 0),
                        GHOST_COLORS[i + 1], d, "Times", size, "bold")
            self.ghost_distance_text.append(t)

    def update_score(self, score):
        """Update the score display."""
        gu.change_text(self.score_text, "SCORE: % 4d" % score)

    def set_team(self, is_blue):
        """Show the team name."""
        text = "RED TEAM"
        if is_blue:
            text = "BLUE TEAM"
        self.team_text = gu.text(self.to_screen(300, 0), self.text_color, text,
                                 "Times", self.font_size, "bold")

    def update_ghost_distances(self, distances):
        """Update the ghost distances display."""
        if len(distances) == 0:
            return
        if 'ghost_distance_text' not in dir(self):
            self.initialize_ghost_distances(distances)
        else:
            for i, d in enumerate(distances):
                gu.change_text(self.ghost_distance_text[i], d)

    # def draw_ghost(self):
    #     pass

    # def draw_pacman(self):
    #     pass

    # def draw_warning(self):
    #     pass

    # def clear_icon(self):
    #     pass

    # def update_message(self, message):
    #     pass

    # def clear_message(self):
    #     pass


class PacmanGraphics:
    """Graphics for Pacman."""

    def __init__(self, zoom=1.0, frame_time=0.0, capture=False):
        """Create graphics object."""
        self.have_window = 0
        self.current_ghost_images = {}
        self.pacman_image = None
        self.zoom = zoom
        self.grid_size = DEFAULT_GRID_SIZE * zoom
        self.capture = capture
        self.frame_time = frame_time

    def check_null_display(self):
        """Return False."""
        return False

    def initialize(self, state, is_blue=False):
        """Initialize graphics from given game state."""
        self.is_blue = is_blue
        self._start_graphics(state)

        # self.draw_distributions(state)
        self.distribution_images = None  # Initialized lazily
        self.draw_static_objects(state)
        self.draw_agent_objects(state)

        # Information
        self.previous_state = state

        # timing
        self.time = time.time()

    def _start_graphics(self, state):
        """Start the graphics by getting layout from state.

        (called from initialize)
        """
        self.layout = state.layout
        layout = self.layout
        self.width = layout.width
        self.height = layout.height
        self.make_window(self.width, self.height)
        self.info_pane = InfoPane(layout, self.grid_size)
        self.current_state = layout

    def draw_distributions(self, state):
        """Draw belief distributions."""
        walls = state.layout.walls
        dist = []
        for x in range(walls.width):
            distx = []
            dist.append(distx)
            for y in range(walls.height):
                (screen_x, screen_y) = self.to_screen((x, y))
                block = gu.square((screen_x, screen_y), 0.5 * self.grid_size,
                                  color=BACKGROUND_COLOR, filled=1, behind=2)
                distx.append(block)
        self.distribution_images = dist

    def draw_static_objects(self, state):
        """Draw walls, food, capsules in state."""
        layout = self.layout
        self.draw_walls(layout.walls)
        self.food = self.draw_food(layout.food)
        self.capsules = self.draw_capsules(layout.capsules)
        gu.refresh()

    def draw_agent_objects(self, state):
        """Draw agents (pacman, ghosts) from state."""
        self.agent_images = []  # (agent_state, image)
        for index, agent in enumerate(state.agent_states):
            if agent.is_pacman:
                image = self.draw_pacman(agent, index)
                self.agent_images.append((agent, image))
            else:
                image = self.draw_ghost(agent, index)
                self.agent_images.append((agent, image))
        gu.refresh()

    def swap_images(self, agent_index, new_state):
        """Change an image from a ghost to a pacman or vice versa.

        Used for capture the flag.
        """
        prev_state, prev_image = self.agent_images[agent_index]
        for item in prev_image:
            gu.remove_from_screen(item)
        if new_state.is_pacman:
            image = self.draw_pacman(new_state, agent_index)
            self.agent_images[agent_index] = (new_state, image)
        else:
            image = self.draw_ghost(new_state, agent_index)
            self.agent_images[agent_index] = (new_state, image)
        gu.refresh()

    def update(self, new_state):
        """Update agents, food, capsules, and info pane from new_state."""
        agent_index = new_state._agent_moved
        agent_state = new_state.agent_states[agent_index]

        if (self.agent_images[agent_index][0].is_pacman !=
                agent_state.is_pacman):
            self.swap_images(agent_index, agent_state)
        prev_state, prev_image = self.agent_images[agent_index]
        if agent_state.is_pacman:
            self.animate_pacman(agent_state, prev_state, prev_image)
        else:
            self.move_ghost(agent_state, agent_index, prev_state, prev_image)
        self.agent_images[agent_index] = (agent_state, prev_image)

        if new_state._food_eaten is not None:
            self.remove_food(new_state._food_eaten, self.food)
        if new_state._capsule_eaten is not None:
            self.remove_capsule(new_state._capsule_eaten, self.capsules)
        self.info_pane.update_score(new_state.score)
        if 'ghost_distances' in dir(new_state):
            self.info_pane.update_ghost_distances(new_state.ghost_distances)

    def make_window(self, width, height):
        """Make window of given size."""
        grid_width = (width - 1) * self.grid_size
        grid_height = (height - 1) * self.grid_size
        screen_width = 2 * self.grid_size + grid_width
        screen_height = 2 * self.grid_size + grid_height + INFO_PANE_HEIGHT

        gu.begin_graphics(screen_width, screen_height, BACKGROUND_COLOR,
                          "CSI-480 Pacman")

    def draw_pacman(self, pacman, index):
        """Draw pacman on screen."""
        position = self.get_position(pacman)
        screen_point = self.to_screen(position)
        endpoints = self.get_endpoints(self.get_direction(pacman))

        width = PACMAN_OUTLINE_WIDTH
        outline_color = PACMAN_COLOR
        fill_color = PACMAN_COLOR

        if self.capture:
            outline_color = TEAM_COLORS[index % 2]
            fill_color = GHOST_COLORS[index]
            width = PACMAN_CAPTURE_OUTLINE_WIDTH

        return [gu.circle(screen_point, PACMAN_SCALE * self.grid_size,
                          fill_color=fill_color, outline_color=outline_color,
                          endpoints=endpoints, width=width)]

    def get_endpoints(self, direction, position=(0, 0)):
        """Get the endpoints for pacman's mouth pie slice."""
        x, y = position
        pos = x - int(x) + y - int(y)
        width = 30 + 80 * math.sin(math.pi * pos)

        delta = width / 2
        if (direction == 'West'):
            endpoints = (180 + delta, 180 - delta)
        elif (direction == 'North'):
            endpoints = (90 + delta, 90 - delta)
        elif (direction == 'South'):
            endpoints = (270 + delta, 270 - delta)
        else:
            endpoints = (0 + delta, 0 - delta)
        return endpoints

    def move_pacman(self, position, direction, image):
        """Move pacman to given position; this is called by animate_pacman."""
        screen_position = self.to_screen(position)
        endpoints = self.get_endpoints(direction, position)
        r = PACMAN_SCALE * self.grid_size
        gu.move_circle(image[0], screen_position, r, endpoints)
        gu.refresh()

    def animate_pacman(self, pacman, prev_pacman, image):
        """Move pacman based on position update and enforce frame rate."""
        if self.frame_time < 0:
            print('Press any key to step forward, "q" to play')
            keys = gu.wait_for_keys()
            if 'q' in keys:
                self.frame_time = 0.1
        if abs(self.frame_time) > 0.00001:
            fx, fy = self.get_position(prev_pacman)
            px, py = self.get_position(pacman)
            frames = 4.0
            for i in range(1, int(frames) + 1):
                pos = (px * i / frames + fx * (frames - i) / frames,
                       py * i / frames + fy * (frames - i) / frames)
                self.move_pacman(pos, self.get_direction(pacman), image)
                gu.refresh()

                elapsed = time.time() - self.time

                if elapsed <= (abs(self.frame_time) / frames):
                    gu.sleep((abs(self.frame_time) / frames) - elapsed)
                self.time = time.time()
        else:
            self.move_pacman(self.get_position(pacman),
                             self.get_direction(pacman), image)
        gu.refresh()

    def get_ghost_color(self, ghost, ghost_index):
        """Determine ghost color based on scared timer."""
        if ghost.scared_timer > 0:
            return SCARED_COLOR
        else:
            return GHOST_COLORS[ghost_index]

    def draw_ghost(self, ghost, agent_index):
        """Draw the given ghost."""
        pos = self.get_position(ghost)
        dir = self.get_direction(ghost)
        (screen_x, screen_y) = (self.to_screen(pos))
        coords = []
        for (x, y) in GHOST_SHAPE:
            coords.append((x * self.grid_size * GHOST_SIZE + screen_x,
                           y * self.grid_size * GHOST_SIZE + screen_y))

        colour = self.get_ghost_color(ghost, agent_index)
        body = gu.polygon(coords, colour, filled=1)

        dx = 0
        dy = 0
        if dir == 'North':
            dy = -0.2
        if dir == 'South':
            dy = 0.2
        if dir == 'East':
            dx = 0.2
        if dir == 'West':
            dx = -0.2
        left_eye = gu.circle(
            (screen_x + self.grid_size * GHOST_SIZE * (-0.3 + dx / 1.5),
             screen_y - self.grid_size * GHOST_SIZE * (0.3 - dy / 1.5)),
            self.grid_size * GHOST_SIZE * 0.2, WHITE, WHITE)
        right_eye = gu.circle(
            (screen_x + self.grid_size * GHOST_SIZE * (0.3 + dx / 1.5),
             screen_y - self.grid_size * GHOST_SIZE * (0.3 - dy / 1.5)),
            self.grid_size * GHOST_SIZE * 0.2, WHITE, WHITE)
        left_pupil = gu.circle(
            (screen_x + self.grid_size * GHOST_SIZE * (-0.3 + dx),
             screen_y - self.grid_size * GHOST_SIZE * (0.3 - dy)),
            self.grid_size * GHOST_SIZE * 0.08, BLACK, BLACK)
        right_pupil = gu.circle(
            (screen_x + self.grid_size * GHOST_SIZE * (0.3 + dx),
             screen_y - self.grid_size * GHOST_SIZE * (0.3 - dy)),
            self.grid_size * GHOST_SIZE * 0.08, BLACK, BLACK)
        ghost_image_parts = []
        ghost_image_parts.append(body)
        ghost_image_parts.append(left_eye)
        ghost_image_parts.append(right_eye)
        ghost_image_parts.append(left_pupil)
        ghost_image_parts.append(right_pupil)

        return ghost_image_parts

    def move_eyes(self, pos, dir, eyes):
        """Move ghost's eyes."""
        (screen_x, screen_y) = (self.to_screen(pos))
        dx = 0
        dy = 0
        if dir == 'North':
            dy = -0.2
        if dir == 'South':
            dy = 0.2
        if dir == 'East':
            dx = 0.2
        if dir == 'West':
            dx = -0.2
        gu.move_circle(
            eyes[0],
            (screen_x + self.grid_size * GHOST_SIZE * (-0.3 + dx / 1.5),
             screen_y - self.grid_size * GHOST_SIZE * (0.3 - dy / 1.5)),
            self.grid_size * GHOST_SIZE * 0.2)
        gu.move_circle(
            eyes[1],
            (screen_x + self.grid_size * GHOST_SIZE * (0.3 + dx / 1.5),
             screen_y - self.grid_size * GHOST_SIZE * (0.3 - dy / 1.5)),
            self.grid_size * GHOST_SIZE * 0.2)
        gu.move_circle(
            eyes[2],
            (screen_x + self.grid_size * GHOST_SIZE * (-0.3 + dx),
             screen_y - self.grid_size * GHOST_SIZE * (0.3 - dy)),
            self.grid_size * GHOST_SIZE * 0.08)
        gu.move_circle(
            eyes[3],
            (screen_x + self.grid_size * GHOST_SIZE * (0.3 + dx),
             screen_y - self.grid_size * GHOST_SIZE * (0.3 - dy)),
            self.grid_size * GHOST_SIZE * 0.08)

    def move_ghost(self, ghost, ghost_index, prev_ghost, ghost_image_parts):
        """Move given ghost."""
        old_x, old_y = self.to_screen(self.get_position(prev_ghost))
        new_x, new_y = self.to_screen(self.get_position(ghost))
        delta = new_x - old_x, new_y - old_y

        for ghost_image_part in ghost_image_parts:
            gu.move_by(ghost_image_part, delta)
        gu.refresh()

        if ghost.scared_timer > 0:
            color = SCARED_COLOR
        else:
            color = GHOST_COLORS[ghost_index]
        gu.edit(ghost_image_parts[0], ('fill', color), ('outline', color))
        self.move_eyes(self.get_position(ghost), self.get_direction(ghost),
                       ghost_image_parts[-4:])
        gu.refresh()

    def get_position(self, agent_state):
        """Get agent's position."""
        if agent_state.configuration is None:
            return (-1000, -1000)
        return agent_state.get_position()

    def get_direction(self, agent_state):
        """Get agent's direction."""
        if agent_state.configuration is None:
            return Directions.STOP
        return agent_state.configuration.get_direction()

    def finish(self):
        """End the graphics."""
        gu.end_graphics()

    def to_screen(self, point):
        """Convert point to screen coordinates."""
        (x, y) = point
        # y = self.height - y
        x = (x + 1) * self.grid_size
        y = (self.height - y) * self.grid_size
        return (x, y)

    def to_screen2(self, point):
        """Convert point to screen coordinats.

        This version Fixes some TK issue with off-center circles.
        """
        (x, y) = point
        # y = self.height - y
        x = (x + 1) * self.grid_size
        y = (self.height - y) * self.grid_size
        return (x, y)

    def draw_walls(self, wall_matrix):
        """Draw the walls."""
        wall_color = WALL_COLOR
        for x_num, x in enumerate(wall_matrix):
            if self.capture and (x_num * 2) < wall_matrix.width:
                wall_color = TEAM_COLORS[0]
            if self.capture and (x_num * 2) >= wall_matrix.width:
                wall_color = TEAM_COLORS[1]

            for y_num, cell in enumerate(x):
                if cell:  # There's a wall here
                    pos = (x_num, y_num)
                    screen = self.to_screen(pos)
                    screen2 = self.to_screen2(pos)

                    # draw each quadrant of the square based on adjacent walls
                    w_is_wall = self.is_wall(x_num - 1, y_num, wall_matrix)
                    e_is_wall = self.is_wall(x_num + 1, y_num, wall_matrix)
                    n_is_wall = self.is_wall(x_num, y_num + 1, wall_matrix)
                    s_is_wall = self.is_wall(x_num, y_num - 1, wall_matrix)
                    nw_is_wall = self.is_wall(x_num - 1, y_num + 1,
                                              wall_matrix)
                    sw_is_wall = self.is_wall(x_num - 1, y_num - 1,
                                              wall_matrix)
                    ne_is_wall = self.is_wall(x_num + 1, y_num + 1,
                                              wall_matrix)
                    se_is_wall = self.is_wall(x_num + 1, y_num - 1,
                                              wall_matrix)

                    # NE quadrant
                    if (not n_is_wall) and (not e_is_wall):
                        # inner circle
                        gu.circle(screen2, WALL_RADIUS * self.grid_size,
                                  wall_color, wall_color, (0, 91), 'arc')
                    if (n_is_wall) and (not e_is_wall):
                        # vertical line
                        gu.line(add(screen, (self.grid_size * WALL_RADIUS, 0)),
                                add(screen, (self.grid_size * WALL_RADIUS,
                                             self.grid_size * (-0.5) - 1)),
                                wall_color)
                    if (not n_is_wall) and (e_is_wall):
                        # horizontal line
                        gu.line(add(screen,
                                    (0, self.grid_size * (-1) * WALL_RADIUS)),
                                add(screen,
                                    (self.grid_size * 0.5 + 1,
                                     self.grid_size * (-1) * WALL_RADIUS)),
                                wall_color)
                    if (n_is_wall) and (e_is_wall) and (not ne_is_wall):
                        # outer circle
                        gu.circle(add(screen2,
                                      (self.grid_size * 2 * WALL_RADIUS,
                                       self.grid_size * (-2) * WALL_RADIUS)),
                                  WALL_RADIUS * self.grid_size - 1, wall_color,
                                  wall_color, (180, 271), 'arc')
                        gu.line(add(screen,
                                    (self.grid_size * 2 * WALL_RADIUS - 1,
                                     self.grid_size * (-1) * WALL_RADIUS)),
                                add(screen,
                                    (self.grid_size * 0.5 + 1,
                                     self.grid_size * (-1) * WALL_RADIUS)),
                                wall_color)
                        gu.line(add(screen,
                                    (self.grid_size * WALL_RADIUS,
                                     self.grid_size * (-2) * WALL_RADIUS + 1)),
                                add(screen, (self.grid_size * WALL_RADIUS,
                                             self.grid_size * (-0.5))),
                                wall_color)

                    # NW quadrant
                    if (not n_is_wall) and (not w_is_wall):
                        # inner circle
                        gu.circle(screen2, WALL_RADIUS * self.grid_size,
                                  wall_color, wall_color, (90, 181), 'arc')
                    if (n_is_wall) and (not w_is_wall):
                        # vertical line
                        gu.line(add(screen,
                                    (self.grid_size * (-1) * WALL_RADIUS, 0)),
                                add(screen,
                                    (self.grid_size * (-1) * WALL_RADIUS,
                                     self.grid_size * (-0.5) - 1)), wall_color)
                    if (not n_is_wall) and (w_is_wall):
                        # horizontal line
                        gu.line(add(screen,
                                    (0, self.grid_size * (-1) * WALL_RADIUS)),
                                add(screen,
                                    (self.grid_size * (-0.5) - 1,
                                     self.grid_size * (-1) * WALL_RADIUS)),
                                wall_color)
                    if (n_is_wall) and (w_is_wall) and (not nw_is_wall):
                        # outer circle
                        gu.circle(add(screen2,
                                      (self.grid_size * (-2) * WALL_RADIUS,
                                       self.grid_size * (-2) * WALL_RADIUS)),
                                  WALL_RADIUS * self.grid_size - 1,
                                  wall_color, wall_color, (270, 361), 'arc')
                        gu.line(add(screen,
                                    (self.grid_size * (-2) * WALL_RADIUS + 1,
                                     self.grid_size * (-1) * WALL_RADIUS)),
                                add(screen,
                                    (self.grid_size * (-0.5),
                                     self.grid_size * (-1) * WALL_RADIUS)),
                                wall_color)
                        gu.line(add(screen,
                                    (self.grid_size * (-1) * WALL_RADIUS,
                                     self.grid_size * (-2) * WALL_RADIUS + 1)),
                                add(screen,
                                    (self.grid_size * (-1) * WALL_RADIUS,
                                     self.grid_size * (-0.5))), wall_color)

                    # SE quadrant
                    if (not s_is_wall) and (not e_is_wall):
                        # inner circle
                        gu.circle(screen2, WALL_RADIUS * self.grid_size,
                                  wall_color, wall_color, (270, 361), 'arc')
                    if (s_is_wall) and (not e_is_wall):
                        # vertical line
                        gu.line(add(screen, (self.grid_size * WALL_RADIUS, 0)),
                                add(screen, (self.grid_size * WALL_RADIUS,
                                             self.grid_size * (0.5) + 1)),
                                wall_color)
                    if (not s_is_wall) and (e_is_wall):
                        # horizontal line
                        gu.line(add(screen,
                                    (0, self.grid_size * (1) * WALL_RADIUS)),
                                add(screen,
                                    (self.grid_size * 0.5 + 1,
                                     self.grid_size * (1) * WALL_RADIUS)),
                                wall_color)
                    if (s_is_wall) and (e_is_wall) and (not se_is_wall):
                        # outer circle
                        gu.circle(add(screen2,
                                      (self.grid_size * 2 * WALL_RADIUS,
                                       self.grid_size * (2) * WALL_RADIUS)),
                                  WALL_RADIUS * self.grid_size - 1, wall_color,
                                  wall_color, (90, 181), 'arc')
                        gu.line(add(screen,
                                    (self.grid_size * 2 * WALL_RADIUS - 1,
                                     self.grid_size * (1) * WALL_RADIUS)),
                                add(screen,
                                    (self.grid_size * 0.5,
                                     self.grid_size * (1) * WALL_RADIUS)),
                                wall_color)
                        gu.line(add(screen,
                                    (self.grid_size * WALL_RADIUS,
                                     self.grid_size * (2) * WALL_RADIUS - 1)),
                                add(screen, (self.grid_size * WALL_RADIUS,
                                             self.grid_size * (0.5))),
                                wall_color)

                    # SW quadrant
                    if (not s_is_wall) and (not w_is_wall):
                        # inner circle
                        gu.circle(screen2,
                                  WALL_RADIUS * self.grid_size, wall_color,
                                  wall_color, (180, 271), 'arc')
                    if (s_is_wall) and (not w_is_wall):
                        # vertical line
                        gu.line(add(screen,
                                    (self.grid_size * (-1) * WALL_RADIUS, 0)),
                                add(screen,
                                    (self.grid_size * (-1) * WALL_RADIUS,
                                     self.grid_size * (0.5) + 1)), wall_color)
                    if (not s_is_wall) and (w_is_wall):
                        # horizontal line
                        gu.line(add(screen,
                                    (0, self.grid_size * (1) * WALL_RADIUS)),
                                add(screen,
                                    (self.grid_size * (-0.5) - 1,
                                     self.grid_size * (1) * WALL_RADIUS)),
                                wall_color)
                    if (s_is_wall) and (w_is_wall) and (not sw_is_wall):
                        # outer circle
                        gu.circle(add(screen2,
                                      (self.grid_size * (-2) * WALL_RADIUS,
                                       self.grid_size * (2) * WALL_RADIUS)),
                                  WALL_RADIUS * self.grid_size - 1, wall_color,
                                  wall_color, (0, 91), 'arc')
                        gu.line(add(screen,
                                    (self.grid_size * (-2) * WALL_RADIUS + 1,
                                     self.grid_size * (1) * WALL_RADIUS)),
                                add(screen,
                                    (self.grid_size * (-0.5),
                                     self.grid_size * (1) * WALL_RADIUS)),
                                wall_color)
                        gu.line(add(screen,
                                    (self.grid_size * (-1) * WALL_RADIUS,
                                     self.grid_size * (2) * WALL_RADIUS - 1)),
                                add(screen,
                                    (self.grid_size * (-1) * WALL_RADIUS,
                                     self.grid_size * (0.5))), wall_color)

    @staticmethod
    def is_wall(x, y, walls):
        """Determine if wall at given coordinate."""
        if x < 0 or y < 0:
            return False
        if x >= walls.width or y >= walls.height:
            return False
        return walls[x][y]

    def draw_food(self, food_matrix):
        """Draw the food dats."""
        food_images = []
        color = FOOD_COLOR
        for x_num, x in enumerate(food_matrix):
            if self.capture and (x_num * 2) <= food_matrix.width:
                color = TEAM_COLORS[0]
            if self.capture and (x_num * 2) > food_matrix.width:
                color = TEAM_COLORS[1]
            image_row = []
            food_images.append(image_row)
            for y_num, cell in enumerate(x):
                if cell:  # There's food here
                    screen = self.to_screen((x_num, y_num))
                    dot = gu.circle(screen, FOOD_SIZE * self.grid_size,
                                    outline_color=color, fill_color=color,
                                    width=1)
                    image_row.append(dot)
                else:
                    image_row.append(None)
        return food_images

    def draw_capsules(self, capsules):
        """Draw capsules (power pellets)."""
        capsule_images = {}
        for capsule in capsules:
            (screen_x, screen_y) = self.to_screen(capsule)
            dot = gu.circle((screen_x, screen_y),
                            CAPSULE_SIZE * self.grid_size,
                            outline_color=CAPSULE_COLOR,
                            fill_color=CAPSULE_COLOR,
                            width=1)
            capsule_images[capsule] = dot
        return capsule_images

    def remove_food(self, cell, food_images):
        """Remove food from the screen in given cell."""
        x, y = cell
        gu.remove_from_screen(food_images[x][y])

    def remove_capsule(self, cell, capsule_images):
        """Remove capsule from the screen in given cell."""
        x, y = cell
        gu.remove_from_screen(capsule_images[(x, y)])

    def draw_expanded_cells(self, cells):
        """Draw an overlay of expanded grid positions for search agents."""
        n = float(len(cells))
        base_color = [1.0, 0.0, 0.0]
        self.clear_expanded_cells()
        self.expanded_cells = []
        for k, cell in enumerate(cells):
            screen_pos = self.to_screen(cell)
            cell_color = gu.format_color(*[(n - k) * c * .5 / n + .25
                                           for c in base_color])
            block = gu.square(screen_pos, 0.5 * self.grid_size,
                              color=cell_color, filled=1, behind=2)
            self.expanded_cells.append(block)
            if self.frame_time < 0:
                gu.refresh()

    def clear_expanded_cells(self):
        """Clear expanded cells (used for search agents)."""
        if 'expanded_cells' in dir(self) and len(self.expanded_cells) > 0:
            for cell in self.expanded_cells:
                gu.remove_from_screen(cell)

    def update_distributions(self, distributions):
        """Draw an agent's belief distributions."""
        # copy all distributions so we don't change their state
        distributions = [x.copy() for x in distributions]
        if self.distribution_images is None:
            self.draw_distributions(self.previous_state)
        for x in range(len(self.distribution_images)):
            for y in range(len(self.distribution_images[0])):
                image = self.distribution_images[x][y]
                weights = [dist[(x, y)] for dist in distributions]

                if sum(weights) != 0:
                    pass
                # Fog of war
                color = [0.0, 0.0, 0.0]
                colors = GHOST_VEC_COLORS[1:]  # With Pacman
                if self.capture:
                    colors = GHOST_VEC_COLORS
                for weight, gcolor in zip(weights, colors):
                    color = [min(1.0, c + 0.95 * g * weight ** .3)
                             for c, g in zip(color, gcolor)]
                gu.change_color(image, gu.format_color(*color))
        gu.refresh()


def add(x, y):
    """Add two coordinate pairs."""
    return (x[0] + y[0], x[1] + y[1])


# Saving graphical output
# -----------------------
# Note: to make an animated gif from this postscript output, try the command:
# convert -delay 7 -loop 1 -compress lzw -layers optimize frame* out.gif
# convert is part of imagemagick (freeware)

SAVE_POSTSCRIPT = False
POSTSCRIPT_OUTPUT_DIR = 'frames'
FRAME_NUMBER = 0


def save_frame():
    """Save the current graphical output as a postscript file."""
    global SAVE_POSTSCRIPT, FRAME_NUMBER, POSTSCRIPT_OUTPUT_DIR
    if not SAVE_POSTSCRIPT:
        return
    if not os.path.exists(POSTSCRIPT_OUTPUT_DIR):
        os.mkdir(POSTSCRIPT_OUTPUT_DIR)
    name = os.path.join(POSTSCRIPT_OUTPUT_DIR, 'frame_%08d.ps' % FRAME_NUMBER)
    FRAME_NUMBER += 1
    gu.write_postscript(name)  # writes the current canvas
