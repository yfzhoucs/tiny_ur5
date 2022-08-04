"""
@author: Olivier Sigaud
A merge between two sources:
* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia
* the gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
# apple: https://unsplash.com/images/food/apple
# orange: https://unsplash.com/s/photos/orange
# wood: https://architextures.org/textures/category/wood
import math
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from gym.utils.renderer import Renderer
import pygame


class TinyUR5Env(gym.Env):
    """
    ### Description
    The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
    at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
    that can be applied to the car in either direction. The goal of the MDP is to strategically
    accelerate the car to reach the goal state on top of the right hill. There are two versions
    of the mountain car domain in gym: one with discrete actions and one with continuous.
    This version is the one with continuous actions.
    This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)
    ```
    @TECHREPORT{Moore90efficientmemory-based,
        author = {Andrew William Moore},
        title = {Efficient Memory-based Learning for Robot Control},
        institution = {University of Cambridge},
        year = {1990}
    }
    ```
    ### Observation Space
    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:
    | Num | Observation                          | Min  | Max | Unit         |
    |-----|--------------------------------------|------|-----|--------------|
    | 0   | position of the car along the x-axis | -Inf | Inf | position (m) |
    | 1   | velocity of the car                  | -Inf | Inf | position (m) |
    ### Action Space
    The action is a `ndarray` with shape `(1,)`, representing the directional force applied on the car.
    The action is clipped in the range `[-1,1]` and multiplied by a power of 0.0015.
    ### Transition Dynamics:
    Given an action, the mountain car follows the following transition dynamics:
    *velocity<sub>t+1</sub> = velocity<sub>t+1</sub> + force * self.power - 0.0025 * cos(3 * position<sub>t</sub>)*
    *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*
    where force is the action clipped to the range `[-1,1]` and power is a constant 0.0015.
    The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall.
    The position is clipped to the range [-1.2, 0.6] and velocity is clipped to the range [-0.07, 0.07].
    ### Reward
    A negative reward of *-0.1 * action<sup>2</sup>* is received at each timestep to penalise for
    taking actions of large magnitude. If the mountain car reaches the goal then a positive reward of +100
    is added to the negative reward for that timestep.
    ### Starting State
    The position of the car is assigned a uniform random value in `[-0.6 , -0.4]`.
    The starting velocity of the car is always assigned to 0.
    ### Episode End
    The episode ends if either of the following happens:
    1. Termination: The position of the car is greater than or equal to 0.45 (the goal position on top of the right hill)
    2. Truncation: The length of the episode is 999.
    ### Arguments
    ```
    gym.make('MountainCarContinuous-v0')
    ```
    ### Version History
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0):
        self.min_action = -np.pi * 2 - 0.01
        self.max_action = np.pi * 2 + 0.01
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = (
            0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        )
        self.goal_velocity = goal_velocity
        self.power = 0.0015

        self.low_state = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)

        self.screen_width = 400
        self.screen_height = 200
        self.screen = None
        self.clock = None
        self.isopen = True

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(4,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(4,), dtype=np.float32
        )

        self.robot_joints = np.zeros((4,), dtype=np.float32)
        self.lim_length = 70

        self.Kp = 15
        self.dt = 0.003


        self.image_lim = pygame.image.load('assets/lim.png')
        self.image_lim = pygame.transform.scale(self.image_lim, (20, 90))

        self.image_gripper_open = pygame.image.load('assets/gripper_open.png')
        self.image_gripper_open = pygame.transform.scale(self.image_gripper_open, (20, 20))

        self.image_gripper_closed = pygame.image.load('assets/gripper_closed.png')
        self.image_gripper_closed = pygame.transform.scale(self.image_gripper_closed, (17, 24))

        self.image_wood = pygame.image.load('assets/wood.jpg')
        self.image_wood = pygame.transform.scale(self.image_wood, (400, 200))

        self.image_apple = pygame.image.load('assets/apple.png')
        self.image_apple = pygame.transform.scale(self.image_apple, (20, 20))

        self.image_orange = pygame.image.load('assets/orange.png')
        self.image_orange = pygame.transform.scale(self.image_orange, (20, 20))

        self.image_banana = pygame.image.load('assets/banana.png')
        self.image_banana = pygame.transform.scale(self.image_banana, (30, 20))


        self.positions = np.array([[360, 120], [40, 120], [80, 180]])

        self.grab = None
        self.grab_position = None


    def _eef_(self):

        start_x = 0
        start_y = 0
        end_x = 200
        end_y = 10
        angle = 0
        for i in range(self.robot_joints.shape[0] - 1):
            if i < 2:
                start_x = end_x
                start_y = end_y
                angle = angle + self.robot_joints[i]
                end_x = start_x + np.sin(angle) * self.lim_length
                end_y = start_y + np.cos(angle) * self.lim_length
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2

            elif i == 2:
                start_x = end_x
                start_y = end_y
                angle = angle + self.robot_joints[i]
                end_x = start_x + np.sin(angle) * 40
                end_y = start_y + np.cos(angle) * 40
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
        return np.array([mid_x, mid_y])

    def _l2_(self, eef, position):
        return ((eef[0] - position[0]) ** 2 + (eef[1] - position[1]) ** 2) ** (1/2)


    def _grab_(self, position, eef):
        grab = (self._l2_(eef, position) < 5)
        return grab


    def _gripper_closed_(self):
        if self.robot_joints[-1] >= 0:
            return True
        else:
            return False


    def step(self, action: np.ndarray):

        # Convert a possible numpy bool to a Python bool.
        terminated = False
        reward = 0

        # action is target angles of the joints
        assert action.shape[0] == self.robot_joints.shape[0]
        for i in range(action.shape[0]):
            self.robot_joints[i] = self.robot_joints[i] + self.Kp * self._ang_diff(action[i], self.robot_joints[i]) * self.dt

        eef = self._eef_()

        if self.grab is not None:
            self.positions[grab_position] = eef + self.grab_position


        self.grab = None
        self.grab_position = None
        if self._gripper_closed_():
            for i in range(len(self.positions)):
                if self._grab_(self.positions[i], eef):
                    self.grab = i
                    self.grab_position = self.positions[i] - eef

        print(self.robot_joints)
        self.renderer.render_step()

        state = {
            'joints': self.robot_joints,
            'positions': self.positions
        }

        return state, reward, terminated, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        self.state = np.array([self.np_random.uniform(low=low, high=high), 0])
        self.renderer.reset()
        self.renderer.render_step()
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    
    def _ang_diff(self, theta1, theta2):
        # Returns the difference between two angles in the range -pi to +pi
        return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi

    def render(self, mode="human"):
        if self.render_mode is not None:
            return self.renderer.get_renders()
        else:
            return self._render(mode)

    def _blitRotate(self, surf, image, origin, pivot, angle):
        image_rect = image.get_rect(topleft = (origin[0] - pivot[0], origin[1]-pivot[1]))
        offset_center_to_pivot = pygame.math.Vector2(origin) - image_rect.center
        rotated_offset = offset_center_to_pivot.rotate(-angle)
        rotated_image_center = (origin[0] - rotated_offset.x, origin[1] - rotated_offset.y)
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_image_rect = rotated_image.get_rect(center = rotated_image_center)
        surf.blit(rotated_image, rotated_image_rect)

    def _render(self, mode="human"):
        assert mode in self.metadata["render_modes"]

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode in {"rgb_array", "single_rgb_array"}
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = 200
        scale = self.screen_width / world_width

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))


        self.surf.blit(self.image_wood, (0, 0))
        self.surf.blit(self.image_apple, self.positions[0])
        self.surf.blit(self.image_orange, self.positions[1])
        self.surf.blit(self.image_banana, self.positions[2])

        start_x = 0
        start_y = 0
        end_x = 200
        end_y = 10
        angle = 0
        for i in range(self.robot_joints.shape[0] - 1):
            if i < 2:
                start_x = end_x
                start_y = end_y
                angle = angle + self.robot_joints[i]
                end_x = start_x + np.sin(angle) * self.lim_length
                end_y = start_y + np.cos(angle) * self.lim_length
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2

                image_lim_transformed = pygame.transform.rotate(self.image_lim, np.rad2deg(angle))
                new_rect = image_lim_transformed.get_rect()
                self.surf.blit(image_lim_transformed, (mid_x - new_rect[2] / 2, mid_y - new_rect[3] / 2))
            elif i == 2:
                start_x = end_x
                start_y = end_y
                angle = angle + self.robot_joints[i]
                end_x = start_x + np.sin(angle) * 40
                end_y = start_y + np.cos(angle) * 40
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2

                print('gripper angle:', self.robot_joints[-1])
                if self.robot_joints[-1] < 0:
                    image_gripper_open_transformed = pygame.transform.rotate(self.image_gripper_open, np.rad2deg(angle))
                    new_rect = image_gripper_open_transformed.get_rect()
                    self.surf.blit(image_gripper_open_transformed, (mid_x - new_rect[2] / 2, mid_y - new_rect[3] / 2))
                else:
                    image_gripper_closed_transformed = pygame.transform.rotate(self.image_gripper_closed, np.rad2deg(angle))
                    new_rect = image_gripper_closed_transformed.get_rect()
                    self.surf.blit(image_gripper_closed_transformed, (mid_x - new_rect[2] / 2, mid_y - new_rect[3] / 2))

        # self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


if __name__ == '__main__':
    # env = TinyUR5Env(render_mode='rgb_array')
    env = TinyUR5Env(render_mode='human')

    observation, info = env.reset(seed=42, return_info=True)

    for i in range(1000):
        print(i)
        input()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        img = env.render()
        # print(img[0].shape)

        if done:
            observation, info = env.reset(return_info=True)
    env.close()
