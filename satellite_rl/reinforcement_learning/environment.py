"""Satellite environment."""
from typing import Optional
import logging

import numpy as np
from gymnasium import Env, spaces

# Define constants
# Modems constants
MOD_NB_LINKS: int = 4
# Maximal monomodal binary rate of the modems (kbps)
MOD_MONO_BIN_RATE: int = 16384
# Maximal multimodal binary rate of the modems (kbps)
MOD_MULTI_BIN_RATE: int = 32768
# Maximal monomodal symbol rate of the modems (kbauds)
MOD_MONO_SYMB_RATE: int = 16384
# Maximal multimodal symbol rate of the modems (kbauds)
MOD_MULTI_SYMB_RATE: int = 16384
# Groups constants
# Maximal number of links within a group
GRP_NB_LINKS: int = 31
# Maximal bandwidth within a group (kHz)
GRP_BANDWIDTH: int = 56000


class SatelliteEnv(Env):
    """Custom environment to model the optimization problem."""

    def __init__(
        self,
        links: list[dict],
        nb_groups_init: int,
        nb_modems_init: int,
        state_init: Optional[np.ndarray] = None,
        nb_modems_per_group: int = GRP_NB_LINKS,
    ) -> None:
        """Initialize the environment.

        Args:
            links (list[dict]): List of links data
            nb_groups_init (int):  Number of groups used in the initial state
            nb_modems_init (int): Number of modems used in the initial state
            nb_modems_per_group (int): Number of modem per group on the initial grid.
            state_init (int): Initial state. If provided, it needs to have the shape
                (nb_links, 2)
        """
        super().__init__()
        self.links = links
        self.nb_links = len(self.links)
        # Initial state variables
        # Initial state if provided
        self.state_init = state_init
        # Number of modems used in the initial state
        self.nb_modems_init = nb_modems_init
        # Number of groups used in the initial state
        self.nb_groups_init = nb_groups_init

        self.nb_modems_per_group = nb_modems_per_group
        # State variables for the current state
        if not isinstance(self.state_init, np.ndarray):
            self.state = np.array([(i, 0) for i in range(self.nb_links)])
        else:
            self.state = np.copy(self.state_init)
            self.check_state_init()
        # Number of modems used in the current state
        self.nb_modems = nb_modems_init
        # Number of groups used in the current state
        self.nb_groups = nb_groups_init
        # An array to keep track of the used groups and modems
        self.groups_modems_array = np.zeros((self.nb_groups, self.nb_modems_per_group), dtype=int)
        # Memorize the optimal state variables
        self.state_min = self.state
        self.nb_modems_min = nb_modems_init
        self.nb_groups_min = nb_groups_init
        # Fill the groups-modems array
        self.update_groups_modems_array()

        # Define spaces
        self.action_shape = (3,)
        self.high_action = np.array(
            [self.nb_links - 1, self.nb_groups_init - 1, self.nb_modems_per_group - 1]
        )
        self.action_space = spaces.Box(
            low=np.zeros(self.action_shape),
            high=self.high_action,
            shape=self.action_shape,
            dtype=int,
        )
        self.observation_shape = (self.nb_links, 2)
        self.high_obs = np.array([self.nb_groups_init - 1, self.nb_modems_per_group - 1])
        self.observation_space = spaces.Box(
            low=np.zeros(self.observation_shape),
            high=np.full(self.observation_shape, self.high_obs),
            shape=self.observation_shape,
            dtype=int,
        )

    def reward_function(self) -> float:
        """Compute the reward."""
        nb_modems = np.sum(self.groups_modems_array)
        nb_groups = np.sum(np.sum(self.groups_modems_array, axis=1) > 0)
        diff = (nb_modems + nb_groups) - (self.nb_groups + self.nb_modems)
        self.nb_modems = nb_modems
        self.nb_groups = nb_groups
        ratio = (self.nb_modems * self.nb_groups) / (self.nb_links * self.nb_links)
        if diff == 0:
            return -((ratio) ** 0.5)
        elif diff < 0:
            return -10 * diff * (1 / ratio) ** 0.5
        elif diff > 0:
            return -diff * (ratio) ** 0.5

    def step(self, action: np.ndarray) -> tuple:
        """Execute one time step within the environment."""
        # Copy variables before action
        state_before_action = np.copy(self.state)
        groups_modems_array_before = np.copy(self.groups_modems_array)
        self.take_action(action)
        legal_move = self.is_legal_move()
        # Reset the environment to the
        # previous state if
        # the action dos not respect the constraints
        if not legal_move:
            self.state = state_before_action
            self.groups_modems_array = groups_modems_array_before
        reward = self.reward_function()
        # Memorize optimal state variables
        if self.nb_modems_min + self.nb_groups_min > self.nb_modems + self.nb_groups:
            self.state_min = np.copy(self.state)
            self.nb_groups_min = self.nb_groups
            self.nb_modems_min = self.nb_modems
        return self.state, reward, False, {}, legal_move

    def take_action(self, action: np.ndarray):
        """Move one link from a case to another one."""
        self.state[action[0], :] = [action[1], action[2]]
        self.update_groups_modems_array()

    def is_legal_move(self) -> bool:
        """Check if the move respects the constraints."""
        modems_ok = self.check_modems()
        groups_ok = self.check_groups()
        return modems_ok and groups_ok

    def update_groups_modems_array(self):
        """Update the groups-modems array.

        This array is used to keep track of the
        used groups and modems. groups_modems_array[i,j] is 1
        if the j-th modem of the i-th group is used and 0 otherwise.
        """
        self.groups_modems_array = np.zeros((self.nb_groups_init, self.nb_modems_init), dtype=int)
        for modem in self.state:
            self.groups_modems_array[modem[0], modem[1]] = 1

    def check_modems(self) -> bool:
        """Check if the modems respect the constraints."""
        for link_coord in self.state:
            indices = np.where(
                np.logical_and(self.state[:, 0] == link_coord[0], self.state[:, 1] == link_coord[1])
            )[0]
            links_in_modem = [self.links[indice] for indice in indices]
            binary_rate = np.sum([link["binary_rate"] for link in links_in_modem])
            symbol_rate = np.sum([link["symbol_rate"] for link in links_in_modem])
            if len(indices) == 1 and (
                binary_rate > MOD_MONO_BIN_RATE or symbol_rate > MOD_MONO_SYMB_RATE
            ):
                return False
            elif (
                len(indices) > MOD_NB_LINKS
                or binary_rate > MOD_MULTI_BIN_RATE
                or symbol_rate > MOD_MULTI_SYMB_RATE
            ):
                return False
        return True

    def check_groups(self) -> bool:
        """Check if the groups respect the constraints."""
        for link_coord in self.state:
            indices = np.where(self.state[:, 0] == link_coord[0])[0]
            links_in_group = [self.links[indice] for indice in indices]
            bandwidth = np.sum([link["bandwidth"] for link in links_in_group])
            inverse_binary_rate = np.sum([link["inverse_binary_rate"] for link in links_in_group])
            min_group_inverse_binary_rate = np.min(
                [link["group_inverse_binary_rate"] for link in links_in_group]
            )
            if (
                bandwidth > GRP_BANDWIDTH
                or len(indices) > GRP_NB_LINKS
                or inverse_binary_rate > min_group_inverse_binary_rate
            ):
                return False
        return True

    def reset(self) -> np.ndarray:
        """Reset the environment to an initial state."""
        if not isinstance(self.state_init, np.ndarray):
            self.state = np.array([(i, 0) for i in range(self.nb_links)])
        else:
            self.state = np.copy(self.state_init)
        self.nb_modems = self.nb_modems_init
        self.nb_groups = self.nb_groups_init
        self.update_groups_modems_array()
        return self.state

    def render(self) -> None:
        """Render the environment."""
        print("State of the environment: ", self.state)

    def check_state_init(self):
        """Check if state_init has the shape (nb_links, 2)."""
        if self.state_init.shape != (self.nb_links, 2):
            raise ValueError(f"Error: state_init must have the shape ({self.nb_links}, 2)")


def greedy_initialisation(links: list) -> SatelliteEnv:
    """Initilialize a rectangle environement greedily."""
    print("Greedy initialization ...")
    env = SatelliteEnv(
        links=links,
        nb_modems_init=len(links),
        nb_groups_init=len(links),
    )
    link_ind = 1
    group_ind = 0
    modem_ind = 1
    # Affect the links to the modems greedily
    while link_ind < env.nb_links:
        _, _, _, _, legal_move = env.step(np.array([link_ind, group_ind, modem_ind]))
        if legal_move:
            link_ind += 1
            modem_ind += 1
        else:
            modem_ind += 1
            if modem_ind > GRP_NB_LINKS:
                modem_ind = 0
                group_ind += 1
    state_init = env.state
    nb_groups = env.nb_groups
    nb_modems = env.nb_modems
    # Look for the group with the most modems
    nb_modems_per_group = np.max(
        [len(np.where(state_init[:, 0] == i)[0]) for i in range(env.nb_links)]
    )
    logging.info(f"Initialize an environment with {nb_groups} groups and {nb_modems} modems")
    return SatelliteEnv(
        links=links,
        nb_groups_init=nb_groups + 1,
        nb_modems_init=nb_modems,
        state_init=state_init,
        nb_modems_per_group=nb_modems_per_group,
    )


def solve_easy_instances(links: list):
    """Solve easy instance with a greedy policy.

    Args:
        links (list): list of links data

    Returns:
        nb_modems_min (int): optimal number of modems
        nb_groups_min (int): otpimal number of groups
    """
    env = SatelliteEnv(links=links, nb_groups_init=len(links), nb_modems_init=len(links))
    link_ind = 1
    modem_ind = 0
    group_ind = 0
    while link_ind < len(links):
        _, _, _, _, legal_move = env.step(np.array([link_ind, group_ind, modem_ind]))
        if legal_move:
            link_ind += 1
        else:
            modem_ind += 1
            if modem_ind > GRP_NB_LINKS - 1:
                modem_ind = 0
                group_ind += 1
    return env.nb_groups_min, env.nb_groups_min
