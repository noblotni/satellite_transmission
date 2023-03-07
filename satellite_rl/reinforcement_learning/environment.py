"""Satellite environment."""
import numpy as np
from gymnasium import Env
from gymnasium import spaces

# Define constants
# Modems constants
MOD_NB_LINKS: int = 4
# Maximal binary rate of the modems (kbps)
MOD_BIN_RATE: int = 16384
# Maximal symbol rate of the modems (kbauds)
MOD_SYMB_RATE: int = 16384
# Groups constants
# Maximal number of links within a group
GRP_NB_LINKS: int = 31
# Maximal bandwidth within a group (kHz)
GRP_BANDWIDTH: int = 56000


class SatelliteEnv(Env):
    """Custom environment to model the optimization problem."""

    def __init__(self, links: list) -> None:
        """Create an environment where to put satellite links.

        Args:
            - links (list): List of links data
        """
        super().__init__()
        self.links: list[dict] = links
        self.nb_links: int = len(self.links)
        self.grp_mod_array: np.ndarray = np.zeros((self.nb_links, self.nb_links))
        # State variables for the current state
        self.state: np.ndarray = np.array([(i, i) for i in range(self.nb_links)])
        # Number of modems
        self.nb_mod: int = self.nb_links
        # Number of groups
        self.nb_grps: int = self.nb_links
        # Memorize the optimal state variables
        self.state_min: np.ndarray = self.state
        self.nb_mod_min: int = self.nb_links
        self.nb_grps_min: int = self.nb_links
        # Fill the group-modem array
        self.update_grp_mod_array()

        self.action_shape: tuple[int] = (3,)
        self.action_space: spaces.Box = spaces.Box(
            low=np.zeros(self.action_shape),
            high=np.full(self.action_shape, self.nb_links - 1),
            shape=self.action_shape,
            dtype=int,
        )
        self.observation_shape: tuple[int, int] = (self.nb_links, 2)
        self.observation_space: spaces.Box = spaces.Box(
            low=np.zeros(self.observation_shape),
            high=np.full(self.observation_shape, self.nb_links),
            shape=self.observation_shape,
            dtype=int,
        )

    def reward_function(self) -> float:
        """Compute the reward."""
        nb_modems: int = np.sum(self.grp_mod_array)
        nb_grps: int = np.sum(np.sum(self.grp_mod_array, axis=1) > 0)
        diff: int = (nb_modems + nb_grps) - (self.nb_grps + self.nb_mod)
        self.nb_mod: int = nb_modems
        self.nb_grps: int = nb_grps
        ratio: float = (self.nb_mod * self.nb_grps) / (self.nb_links * self.nb_links)
        if diff == 0:
            return -((ratio) ** 0.5)
        elif diff < 0:
            return -10 * diff * (1 / ratio) ** 0.5
        elif diff > 0:
            return -diff * (ratio) ** 0.5

    def step(self, action: np.ndarray) -> tuple[np.array, float, bool, dict]:
        """Execute one time step within the environment."""
        # Copy variables before action
        state_before_action: np.ndarray = np.copy(self.state)
        grp_mod_array_before: np.ndarray = np.copy(self.grp_mod_array)
        self.take_action(action)
        # Reset the environment
        # previous state if
        # the action dos not respect the constraints
        if not self.is_legal_move():
            self.state = state_before_action
            self.grp_mod_array = grp_mod_array_before
        reward: float = self.reward_function()
        # Memorize optimal state variables
        has_improved: bool = self.nb_mod_min + self.nb_grps_min > self.nb_mod + self.nb_grps
        if has_improved:
            self.state_min = np.copy(self.state)
            self.nb_grps_min = self.nb_grps
            self.nb_mod_min = self.nb_mod
        return self.state, reward, False, {}

    def take_action(self, action: np.ndarray) -> None:
        """Move one link from a case to another one."""
        self.state[action[0], :] = [action[1], action[2]]
        self.update_grp_mod_array()

    def is_legal_move(self) -> bool:
        """Check if the move respects the constraints."""
        are_modems_ok: bool = self.check_modems()
        are_groups_ok: bool = self.check_groups()
        return are_modems_ok and are_groups_ok

    def update_grp_mod_array(self) -> None:
        self.grp_mod_array: np.ndarray = np.zeros((self.nb_links, self.nb_links))
        for modem in self.state:
            self.grp_mod_array[modem[0], modem[1]] = 1

    def check_modems(self) -> bool:
        """Check if the modems respect the constraints."""
        for s in self.state:
            indices: np.ndarray[int] = np.where(
                np.logical_and(self.state[:, 0] == s[0], self.state[:, 1] == s[1])
            )[0]
            links_in_modem: list[dict] = [self.links[indice] for indice in indices]
            binary_rate: int = np.sum([link["binary_rate"] for link in links_in_modem])
            symbol_rate: float = np.sum([link["symbol_rate"] for link in links_in_modem])
            if (
                    len(indices) > MOD_NB_LINKS
                    or binary_rate > MOD_BIN_RATE
                    or symbol_rate > MOD_SYMB_RATE
            ):
                return False
        return True

    def check_groups(self) -> bool:
        """Check if the groups respect the constraints."""
        for s in self.state:
            indices: int = np.where(self.state[:, 0] == s[0])[0]
            links_in_group: list = [self.links[indice] for indice in indices]
            bandwidth: float = np.sum([link["bandwidth"] for link in links_in_group])
            inverse_binary_rate: int = np.sum(
                [link["inverse_binary_rate"] for link in links_in_group]
            )
            min_group_inverse_binary_rate: float = np.min(
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
        self.state: np.ndarray = np.array([(i, i) for i in range(self.nb_links)])
        self.nb_mod: int = self.nb_links
        self.nb_grps: int = self.nb_links
        self.update_grp_mod_array()
        return self.state

    def render(self) -> None:
        """Render the environment."""
        print("State of the environment: ", self.state)
