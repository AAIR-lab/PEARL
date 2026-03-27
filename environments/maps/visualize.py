import os
import sys
import numpy as np
import random 
import copy
import gymnasium as gym
import shapely
import map_maker as map_maker

try:
    import pygame
except ImportError:
    print('Pygame not available ')


class RenderView:
    """ This class displays a :class:`PinballModel`

    This class is used in conjunction with the :func:`run_pinballview`
    function, acting as a *controller*.

    We use `pygame <http://www.pygame.org/>` to draw the environment.

    """
    def __init__(self, screen, obs_points, ball_pos, coffee_pos, mail_pos, target_pos):
        """
        :param screen: a pygame surface
        :type screen: :class:`pygame.Surface`
        :param model: an instance of a :class:`PinballModel`
        :type model: :class:`PinballModel`
        """
        self.screen = screen

        self.DARK_GRAY = [64, 64, 64]
        self.LIGHT_GRAY = [232, 232, 232]
        self.BALL_COLOR = [0, 0, 255]
        self.BALL_COLOR_coffee = [0, 255, 255]
        self.BALL_COLOR_mail = [255, 255, 0]
        self.BALL_COLOR_coffee_mail = [0, 255, 0]
        self.COFFEE_COLOR = [0, 255, 255]
        self.MAIL_COLOR = [255, 255, 0]
        self.TARGET_COLOR = [255, 0, 0]
        self.coffee_pos = coffee_pos
        self.ball_pos = ball_pos
        self.mail_pos = mail_pos

        self.obs_points = obs_points
        self.target_pos = target_pos

        # Draw the background
        self.screen.fill((255, 255, 255))

        for points in self.obs_points:
            points = [self._to_pixels(o) for o in points]
            pygame.draw.polygon(self.screen, self.DARK_GRAY, points, 0)

        pygame.draw.circle(self.screen, self.BALL_COLOR, self._to_pixels(ball_pos), int(0.02*self.screen.get_width()))
        if self.coffee_pos is not None:
            pygame.draw.circle(self.screen, self.COFFEE_COLOR, self._to_pixels(self.coffee_pos), int(0.02*self.screen.get_width()))
        if self.mail_pos is not None:
            pygame.draw.circle(self.screen, self.MAIL_COLOR, self._to_pixels(self.mail_pos), int(0.02*self.screen.get_width()))
        pygame.draw.circle(self.screen, self.TARGET_COLOR, self._to_pixels(self.target_pos), int(0.02*self.screen.get_width()))

        pygame.display.flip()


    def _to_pixels(self, pt):
        """ Converts from real units in the 0-1 range to pixel units

        :param pt: a point in real units
        :type pt: list
        :returns: the input point in pixel units
        :rtype: list

        """
        return [int(pt[0] * self.screen.get_width()), int(pt[1] * self.screen.get_height())]

    def blit(self, agent_pos, has_coffee, has_mail):
        """ Blit the ball onto the background surface """
        self.screen.fill((255, 255, 255))
        for points in self.obs_points:
            points = [self._to_pixels(o) for o in points]
            pygame.draw.polygon(self.screen, self.DARK_GRAY, points, 0)
        pygame.draw.circle(self.screen, self.TARGET_COLOR, self._to_pixels(self.target_pos), int(0.02*self.screen.get_width()))
        if self.coffee_pos is not None:
            pygame.draw.circle(self.screen, self.COFFEE_COLOR, self._to_pixels(self.coffee_pos), int(0.02*self.screen.get_width()))
        if self.mail_pos is not None:
            pygame.draw.circle(self.screen, self.MAIL_COLOR, self._to_pixels(self.mail_pos), int(0.02*self.screen.get_width()))
        if has_coffee and not has_mail:
            ball_color = self.BALL_COLOR_coffee
        elif has_mail and not has_coffee:
            ball_color = self.BALL_COLOR_mail
        elif has_coffee and has_mail:
            ball_color = self.BALL_COLOR_coffee_mail
        else:
            ball_color = self.BALL_COLOR
        pygame.draw.circle(self.screen, ball_color, self._to_pixels(agent_pos), int(0.02*self.screen.get_width()))
        pygame.display.flip()

    def render_and_save(self, img_path):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.blit(self.ball_pos, 0, 0)
        pygame.display.flip()

        # Save the screen as an image file
        pygame.image.save(self.screen, img_path)

    def close(self):
        pygame.quit()


# map_name = "office_harder.cfg"
# start, coffee, mail, goal = (0.05, 0.05), (0.1, 0.8), (0.54, 0.55), (0.95, 0.95)
# _init_state = list(start) + [0,0]
# _goal_state = list(goal) + [1,1]

# map_name = "city1.cfg"
# _init_state = [0.2, 0.7]
# _goal_state = [0.9, 0.35]

# map_name = "city2.cfg"
# _init_state = [0.5, 0.7]
# _goal_state = [0.2, 0.25]


map_name = "city3.cfg"
_init_state = [0.7, 0.35]
_goal_state = [0.2, 0.7]

coffee = None
mail = None

basepath = os.getcwd()
configuration = f"{basepath}/"+map_name
obstacles, points = map_maker.read_obstacles(configuration)
img_path = f"{basepath}/{map_name[:-4]}.png"

pygame.init()
pygame.display.set_caption('Office Domain')
width, height = 500, 500
screen = pygame.display.set_mode((width, height))
environment_view = RenderView(screen, points, _init_state, coffee, mail, _goal_state)
environment_view.render_and_save(img_path)
environment_view.close()