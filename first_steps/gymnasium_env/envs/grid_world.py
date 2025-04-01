import os
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import matplotlib.pyplot as plt


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, size=10):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.hit=False
        self.last_direction=(0,0)
        self.steps_taken=0
        self.reward_tracker=[]

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "barrier": spaces.Box(0, size - 1, shape=(10,2), dtype=int)
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: ([1, 0]),
            Actions.up.value: ([0, 1]),
            Actions.left.value: ([-1, 0]),
            Actions.down.value: ([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": tuple(self._agent_location), "target": tuple(self._target_location), "barrier": tuple(self._barriers)}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                np.array(self._agent_location) - np.array(self._target_location), ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.steps_taken=0

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        self._barriers=[]
        for _ in range(self.observation_space['barrier'].shape[0]):
            self._barrier_location = self.np_random.integers(0, self.size, size=2, dtype=int)

            while np.array_equal(self._agent_location,self._barrier_location) or np.array_equal(self._target_location,self._barrier_location) or any(np.array_equal(self._barrier_location, barrier) for barrier in self._barriers):
                self._barrier_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            self._barriers.append(self._barrier_location)

        self._barriers = np.array(self._barriers)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.steps_taken+=1
        reward=-0.05*np.emath.logn(100,self.steps_taken)
        self.hit=False
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        action = int(action)
        direction = self._action_to_direction[action]
        self.last_direction = direction
        new_agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)
        if any(np.array_equal(new_agent_location, barrier) for barrier in self._barriers):
            reward -= 0.25  # Small penalty for hitting a barrier
            self.hit=True
        else:
            self._agent_location = new_agent_location

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = max(reward+100,0) if terminated else reward  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        old_distance = np.sum(self._agent_location - self._target_location)
        new_distance = np.sum(new_agent_location - self._target_location)
        reward += (old_distance - new_distance) * 0.2  # Positive reward if moving closer, will not hurt to move away
        self.reward_tracker.append(reward)
        print(f'Reward this step: {reward}, distance = {max((old_distance - new_distance) * 0.4, 0)}, mean={np.mean(self.reward_tracker)}')
        if self.steps_taken%10==0:
            plt.hist(self.reward_tracker, bins=20)
            plt.xlabel("Reward Value")
            plt.ylabel("Frequency")
            plt.title("Reward Distribution per Step")
            plt.show(block=False)
            plt.pause(0.001)
            
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Window")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Next we draw the barrier blocks
        for barrier_location in self._barriers:
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    pix_square_size * barrier_location,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Next, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        # Now we draw the agent
        if self.hit:
            pygame.draw.circle(
                canvas,
                (100,100,100),
                ((self._agent_location+0.5)+np.array(self.last_direction)*0.5)*pix_square_size,
                pix_square_size/3
            )
            pygame.draw.circle(
                canvas,
                (255, 122, 122),
                (self._agent_location + 0.5) * pix_square_size,
                pix_square_size / 3,
            )
        else:
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (self._agent_location + 0.5) * pix_square_size,
                pix_square_size / 3,
            )




        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
