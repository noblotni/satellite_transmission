"""Satellite environment."""
from gym import Env
from gym import spaces
import numpy as np

# Define constants
# Modems constants
MOD_NB_LINKS = 4
# Maximal binary rate of the modems (kbps)
MOD_BIN_RATE = 16384
# Maximal symbol rate of the modems (kbauds)
MOD_SYMB_RATE = 16384
# Groups constants
# Maximal number of links within a group
GRP_NB_LINKS = 31
# Maximal bandwidth within a group (kHz)
GRP_BANDWIDTH = 56000


class SatelliteEnv(Env):
    """Custom environment to model the optimization problem."""

    def __init__(self, links: list):
        """Create an environment where to put satellite links.

        Args:
            - links (list): List of links data
        """
        super().__init__
        self.links = links
        self.nb_links = len(self.links)
        self.grp_mod_array = np.zeros((self.nb_links, self.nb_links))
        self.state = np.array([(i, i) for i in range(self.nb_links)])
        # Memorize the sum of the number of modems and the number of groups
        # for the current state
        self.sum_mod_groups = 2 * self.nb_links
        # Memorize the optimal sum and state
        self.state_min = self.state
        self.sum_mod_groups_min = 2 * self.nb_links
        # Fill the group-modem array
        for (i, s) in enumerate(self.state):
            self.grp_mod_array[i, s[0], s[1]] = 1
        self.action_shape = (3,)
        self.action_space = spaces.Box(
            low=np.zeros(self.action_shape),
            high=np.full(self.action_shape, self.nb_links - 1),
            shape=self.action_shape,
            dtype=int,
        )
        self.observation_shape = (self.nb_links, 2)
        self.observation_space = spaces.Box(
            low=np.zeros(self.observation_shape),
            high=np.full(self.observation_shape, self.nb_links),
            shape=self.observation_shape,
            dtype=int,
        )

    def reward_function(self) -> float:
        """Compute the reward."""
        nb_modems = np.sum(self.grp_mod_array)
        nb_grps = np.sum(np.sum(self.grp_mod_array, axis=1) > 0)
        diff = (nb_modems + nb_grps) - self.sum_mod_groups
        self.sum_mod_groups = nb_modems + nb_grps
        if diff == 0:
            return -1
        elif diff < 0:
            return -10 * diff
        elif diff > 0:
            return -diff

    def step(self, action: np.ndarray) -> tuple:
        """Execute one time step within the environment."""
        state_before_action = np.copy(self.state)
        grp_mod_array_before = np.copy(self.grp_mod_array)
        self.take_action(action)
        legal_move = self.is_legal_move()
        if not legal_move:
            self.state = state_before_action
            self.grp_mod_array = grp_mod_array_before
        reward = self.reward_function()
        # Memorize state and sum min
        if self.sum_mod_groups < self.sum_mod_groups_min:
            self.state_min = self.state
            self.sum_mod_groups_min = self.sum_mod_groups_min
        return self.state, reward, False, {}

    def take_action(self, action: np.ndarray):
        """Move one link from a case to another one."""
        state_before = self.state[action[0], :]
        self.grp_mod_array[state_before[0], state_before[1]] = 0
        self.state[action[0], :] = [action[1], action[2]]
        self.grp_mod_array[action[0], action[1], action[2]] = 1

    def is_legal_move(self) -> bool:
        """Check if the move respects the constraints."""
        modems_ok = self.check_modems()
        groups_ok = self.check_groups()
        return modems_ok and groups_ok

    def check_modems(self) -> bool:
        """Check if the modems respect the constraints."""
        duplicate = []
        for s in self.state:
            if not np.any(s == duplicate):  # if s not in np.array(duplicate):
                indices = set(np.where(self.state == s)[0])
                links_indices = [self.links[indice] for indice in indices]
                binary_flow = np.sum([link["binary_rate"] for link in links_indices])
                symbol_rate = np.sum([link["symbol_rate"] for link in links_indices])
            if (
                binary_flow > MOD_BIN_RATE
                or symbol_rate > MOD_SYMB_RATE
                or len(indices) > MOD_NB_LINKS
            ):
                return False
            duplicate.append(s)
        return True

    def check_groups(self) -> bool:
        """Check if the groups respect the constraints."""
        duplicate = []
        for s in self.state:
            if not np.any(s == duplicate):
                indices = set(np.where(self.state[:, 0] == s[0])[0])
                links_indices = [self.links[indice] for indice in indices]
                bandwidth = np.sum([link["bandwidth"] for link in links_indices])
                inverse_binary_rate = np.sum(
                    [link["inverse_binary_rate"] for link in links_indices]
                )
                min_group_inverse_binary_rate = np.min(
                    [link["group_inverse_binary_rate"] for link in links_indices]
                )
            if (
                bandwidth > GRP_BANDWIDTH
                or len(indices) > GRP_NB_LINKS
                or inverse_binary_rate > min_group_inverse_binary_rate
            ):
                return False
            duplicate.append(s)
        return True

    def reset(self) -> np.ndarray:
        """Reset the environment to an initial state."""
        self.state = np.array([(i, i) for i in range(self.nb_links)])
        self.grp_mod_array = np.zeros((self.nb_links, self.nb_links))
        self.sum_mod_groups = 2 * self.nb_links
        for s in self.state:
            self.grp_mod_array[s[0], s[1]] = 1
        return self.state

    def render(self) -> None:
        """Render the environment."""
        print("State of the environment: ", self.state)
