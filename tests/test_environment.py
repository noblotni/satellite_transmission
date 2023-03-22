"""Test the environment."""
import json
from pathlib import Path
from satellite_rl.reinforcement_learning.environment import (
    SatelliteEnv,
    greedy_initialisation,
    solve_easy_instances,
)
import numpy as np


def test_simple_init():
    toy_datafile = Path("./tests/test_data/toy_instance.json")
    with open(toy_datafile, "r", encoding="utf-8") as file:
        links = json.load(file)
    env = SatelliteEnv(links=links, nb_modems_init=len(links), nb_groups_init=len(links))
    assert (env.state == np.array([(i, 0) for i in range(len(links))])).all()


def test_greedy_init():
    toy_datafile = Path("./tests/test_data/toy_instance.json")
    with open(toy_datafile, "r", encoding="utf-8") as file:
        links = json.load(file)
    env = greedy_initialisation(links)
    assert (env.state == np.array([(0, 0), (0, 1), (0, 2), (0, 3)])).all()


def test_solve_easy_instance():
    toy_datafile = Path("./tests/test_data/easy_instance.json")
    with open(toy_datafile, "r", encoding="utf-8") as file:
        links = json.load(file)
    nb_modems_min, nb_groups_min = solve_easy_instances(links)
    assert nb_modems_min == 50
    assert nb_groups_min == 50
