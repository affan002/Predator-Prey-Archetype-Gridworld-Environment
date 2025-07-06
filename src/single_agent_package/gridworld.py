import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    # super().reset()

    def __init__(self, render_mode=None, size=5, perc_num_obstacle = 30 ):

        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.perc_num_obstacle = perc_num_obstacle
        self._num_obstacles = int((self.perc_num_obstacle/100)*(self.size*self.size))

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).

        self.observation_space = self._make_observation_space() # implement self._make_observation_space()

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = self._make_action_space() # implement self._make_action_space()

        self._action_to_direction = self._action_to_direction() # implement self._action_to_direction()

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


       
    def _make_observation_space(self):
        observation_space = spaces.Dict(
                {
                    "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                    "target": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                }
            )
        return observation_space

    def _make_action_space(self):
        action_space = spaces.Discrete(4)
        return action_space
    

    def _action_to_direction(self):
        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        action_to_direction = {
                0: np.array([1, 0]),
                1: np.array([0, 1]),
                2: np.array([-1, 0]),
                3: np.array([0, -1]),}

        return action_to_direction
    

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }


    def reset(self, seed=None, options=None,start_location="random",target_location="random"):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._start_location = self._initialize_start_location(loc=start_location) # implement _initialize_start_obstacle()

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._initialize_target_location(loc=target_location) # implement _initialize_target_obstacle()
        while np.array_equal(self._start_location, self._target_location):
            self._target_location = self._initialize_target_location(loc=target_location)

        self._agent_location = self._start_location

        self._obstacle_location = self._initialize_obstacle() # implement _initialize_obstacle()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    

    def _initialize_start_location(self,loc="random"):
        if loc=='random':
            start_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        else:
            start_location = np.array(loc)

        return start_location
    
    def _initialize_target_location(self,loc="random"):
        if loc=='random':
            target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        else:
            target_location = np.array(loc)

        return target_location
    
    def _initialize_obstacle(self):
        obstacle_location = []
        for i in range(self._num_obstacles):
            temp = self.np_random.integers(0, self.size, size=2, dtype=int)
            cond1 = np.array_equal(temp, self._start_location)
            cond2 = np.array_equal(temp, self._target_location)
            # cond3 = any(np.array_equal(temp, arr) for arr in self._obstacle_location)
            while  cond1 or cond2:
                temp = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
            obstacle_location.append(temp)
        return obstacle_location
    

    def _reward_system(self,agent_location):
        agent_win = np.array_equal(self._agent_location, self._target_location)
        obstacle_hit = any(np.array_equal(self._agent_location, arr) for arr in self._obstacle_location)

        # reward logic
        if obstacle_hit:
            reward = -1
        elif agent_win:
            reward = 1
        else:
            reward = 0

        return reward


    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        reward = self._reward_system(self._agent_location) # implement _reward_system()

        observation = self._get_obs()

        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    

    def render(self):
        if self.render_mode == "rgb_array":
                return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
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
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # First we draw the start
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._start_location,
                (pix_square_size, pix_square_size),
            ),
        )

            # First we draw the obstacles
        for i in range(self._num_obstacles):
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect( # Rect(left, top, width, height) -> Rect
                    pix_square_size * self._obstacle_location[i], #loc
                    (pix_square_size, pix_square_size), # size
                ),
            )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
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

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def render(self):
        if self.render_mode == "rgb_array":
                return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()