from datetime import datetime

import numpy as np
from termcolor import colored
from tqdm import tqdm

from satellite_rl.reinforcement_learning.agents.actor_critic import run_actor_critic
from satellite_rl.reinforcement_learning.agents.ppo import run_ppo


def batch_comparison(
    links,
    algo,
    nb_episodes,
    duration_episode,
    nb_repeat,
    print_freq,
    log_freq,
    timeout,
    verbose,
    generate_report,
    filename,
):
    """
    This function performs a comparison between reinforcement learning algorithms.

    Args:
        links (list): The list of link capacities.
        algo (str): The algorithm to compare. It can be either "actor-critic", "ppo", or "compare".
        nb_episodes (int): The number of episodes for each algorithm.
        duration_episode (int): The duration of each episode for each algorithm.
        nb_repeat (int): The number of times to repeat the experiment.
        print_freq (int): The frequency of printing progress messages.
        log_freq (int): The frequency of logging results.
        timeout (int): The timeout in seconds for each algorithm.
        verbose (int): The level of verbosity. 0 for quiet mode, 1 for normal mode, 2 for verbose mode.
        generate_report (bool): Whether to generate a report.
        filename (str): The name of the file to save the results.

    Returns:
        nb_grps_min_list (list): The list of the minimum number of groups for each algorithm.
        nb_mod_min_list (list): The list of the minimum number of modules for each algorithm.
        state_min_list (list): The list of the minimum states for each algorithm.
    """
    print("=========================================")
    print("Starting at {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    print("=========================================")
    nb_grps_min_list = []
    nb_mod_min_list = []
    state_min_list = []

    if algo == "compare":
        nb_grps_min_list_actor = []
        nb_mod_min_list_actor = []
        state_min_list_actor = []
        nb_grps_min_list_ppo = []
        nb_mod_min_list_ppo = []
        state_min_list_ppo = []

    best_ppo = 0
    best_actor = 0
    # verbose = 0 if nb_repeat > 1 else 1

    if verbose == 0:
        for i in tqdm(range(nb_repeat)):
            if algo == "actor-critic":
                state_min, nb_grps_min, nb_mod_min = run_actor_critic(
                    links,
                    nb_episodes=nb_episodes,
                    duration_episode=duration_episode,
                    print_freq=print_freq,
                    log_freq=log_freq,
                    timeout=timeout,
                    verbose=verbose,
                    report=generate_report,
                    filename=filename + "-" + str(i),
                    batch=True,
                    compare=False,
                )
                state_min_list.append(state_min)
                nb_grps_min_list.append(nb_grps_min)
                nb_mod_min_list.append(nb_mod_min)

            elif algo == "ppo":
                state_min, nb_grps_min, nb_mod_min = run_ppo(
                    links,
                    nb_episodes=nb_episodes,
                    duration_episode=duration_episode,
                    print_freq=print_freq,
                    log_freq=log_freq,
                    timeout=timeout,
                    verbose=verbose,
                    report=generate_report,
                    filename=filename + "-" + str(i),
                    batch=True,
                    compare=False,
                )
                state_min_list.append(state_min)
                nb_grps_min_list.append(nb_grps_min)
                nb_mod_min_list.append(nb_mod_min)

            elif algo == "compare":
                (
                    state_min_actor,
                    nb_grps_min_actor,
                    nb_mod_min_actor,
                ) = run_actor_critic(
                    links,
                    nb_episodes=nb_episodes,
                    duration_episode=1000,
                    print_freq=print_freq,
                    log_freq=log_freq,
                    timeout=timeout,
                    verbose=verbose,
                    report=generate_report,
                    filename=filename + "-" + str(i),
                    batch=True,
                    compare=True,
                )
                state_min_ppo, nb_grps_min_ppo, nb_mod_min_ppo = run_ppo(
                    links,
                    nb_episodes=nb_episodes,
                    duration_episode=13000,
                    print_freq=print_freq,
                    log_freq=log_freq,
                    timeout=timeout,
                    verbose=verbose,
                    report=generate_report,
                    filename=filename + "-" + str(i),
                    batch=True,
                    compare=True,
                )
                nb_grps_min_list_actor.append(nb_grps_min_actor)
                nb_mod_min_list_actor.append(nb_mod_min_actor)
                state_min_list_actor.append(state_min_actor)

                nb_grps_min_list_ppo.append(nb_grps_min_ppo)
                nb_mod_min_list_ppo.append(nb_mod_min_ppo)
                state_min_list_ppo.append(state_min_ppo)

                if nb_grps_min_ppo + nb_mod_min_ppo < nb_grps_min_actor + nb_mod_min_actor:
                    best_ppo += 1
                else:
                    best_actor += 1
    else:
        for i in range(nb_repeat):
            print(f"Repeat {i+1}/{nb_repeat}...")
            if algo == "actor-critic":
                state_min, nb_grps_min, nb_mod_min = run_actor_critic(
                    links,
                    nb_episodes=nb_episodes,
                    duration_episode=duration_episode,
                    print_freq=print_freq,
                    log_freq=log_freq,
                    timeout=timeout,
                    verbose=verbose,
                    report=generate_report,
                    filename=filename + "-" + str(i),
                    batch=True,
                    compare=False,
                )
                state_min_list.append(state_min)
                nb_grps_min_list.append(nb_grps_min)
                nb_mod_min_list.append(nb_mod_min)

            elif algo == "ppo":
                state_min, nb_grps_min, nb_mod_min = run_ppo(
                    links,
                    nb_episodes=nb_episodes,
                    duration_episode=duration_episode,
                    print_freq=print_freq,
                    log_freq=log_freq,
                    timeout=timeout,
                    verbose=verbose,
                    report=generate_report,
                    filename=filename + "-" + str(i),
                    batch=True,
                    compare=False,
                )
                state_min_list.append(state_min)
                nb_grps_min_list.append(nb_grps_min)
                nb_mod_min_list.append(nb_mod_min)

            elif algo == "compare":
                print("Actor-Critic...")
                (
                    state_min_actor,
                    nb_grps_min_actor,
                    nb_mod_min_actor,
                ) = run_actor_critic(
                    links,
                    nb_episodes=nb_episodes,
                    duration_episode=1000,
                    print_freq=print_freq,
                    log_freq=log_freq,
                    timeout=timeout,
                    verbose=verbose,
                    report=generate_report,
                    filename=filename + "-" + str(i),
                    batch=True,
                    compare=True,
                )
                print("PPO...")
                state_min_ppo, nb_grps_min_ppo, nb_mod_min_ppo = run_ppo(
                    links,
                    nb_episodes=nb_episodes,
                    duration_episode=13000,
                    print_freq=print_freq,
                    log_freq=log_freq,
                    timeout=timeout,
                    verbose=verbose,
                    report=generate_report,
                    filename=filename + "-" + str(i),
                    batch=True,
                    compare=True,
                )
                nb_grps_min_list_actor.append(nb_grps_min_actor)
                nb_mod_min_list_actor.append(nb_mod_min_actor)
                state_min_list_actor.append(state_min_actor)

                nb_grps_min_list_ppo.append(nb_grps_min_ppo)
                nb_mod_min_list_ppo.append(nb_mod_min_ppo)
                state_min_list_ppo.append(state_min_ppo)

                if nb_grps_min_ppo + nb_mod_min_ppo < nb_grps_min_actor + nb_mod_min_actor:
                    best_ppo += 1
                else:
                    best_actor += 1

    print("=========================================")
    print("RESULTS")
    print("=========================================")
    if algo == "compare":
        print(colored("COMPARISON BETWEEN PPO AND ACTOR-CRITIC", "blue"))
        print("=========================================")
        if best_ppo > best_actor:
            print(
                colored("PPO", "green"),
                "is better most of the times",
                colored(f"{best_ppo = }", "green"),
                colored(f"{best_actor = }", "red"),
            )
        else:
            print(
                colored("Actor-Critic", "green"),
                "is better most of the times",
                colored(f"{best_actor = }", "green"),
                colored(f"{best_ppo = }", "red"),
            )

        nb_grps_min_list_ppo = np.array(nb_grps_min_list_ppo)
        nb_mod_min_list_ppo = np.array(nb_mod_min_list_ppo)
        nb_grps_min_list_actor = np.array(nb_grps_min_list_actor)
        nb_mod_min_list_actor = np.array(nb_mod_min_list_actor)

        best_ppo_index = np.argmin(nb_grps_min_list_ppo + nb_mod_min_list_ppo)
        state_min_ppo = state_min_list_ppo[best_ppo_index]
        nb_grps_min_ppo = nb_grps_min_list_ppo[best_ppo_index]
        nb_mod_min_ppo = nb_mod_min_list_ppo[best_ppo_index]

        best_actor_index = np.argmin(nb_grps_min_list_actor + nb_mod_min_list_actor)
        state_min_actor = state_min_list_actor[best_actor_index]
        nb_grps_min_actor = nb_grps_min_list_actor[best_actor_index]
        nb_mod_min_actor = nb_mod_min_list_actor[best_actor_index]

        print(colored("Best PPO solution:", "blue"))
        print(f"nb_grps_min_ppo={colored(nb_grps_min_ppo, 'yellow')}")
        print(f"nb_mod_min_ppo={colored(nb_mod_min_ppo, 'yellow')}")

        print(colored("Best Actor-Critic solution:", "blue"))
        print(f"nb_grps_min_actor={colored(nb_grps_min_actor, 'yellow')}")
        print(f"nb_mod_min_actor={colored(nb_mod_min_actor, 'yellow')}")

        print("=========================================")
        if nb_grps_min_ppo + nb_mod_min_ppo < nb_grps_min_actor + nb_mod_min_actor:
            print(colored("PPO", "green"), "found the best solution")
            state_min = state_min_ppo
            nb_grps_min = nb_grps_min_ppo
            nb_mod_min = nb_mod_min_ppo
        else:
            print(colored("Actor-Critic", "green"), "found the best solution")
            state_min = state_min_actor
            nb_grps_min = nb_grps_min_actor
            nb_mod_min = nb_mod_min_actor
    else:
        best_index = np.argmin(nb_grps_min_list + nb_grps_min_list)
        state_min = state_min_list[best_index]
        nb_grps_min = nb_grps_min_list[best_index]
        nb_mod_min = nb_mod_min_list[best_index]

    if algo == "ppo" or algo == "actor-critic":
        return state_min, nb_grps_min, nb_mod_min
    else:
        return state_min, nb_grps_min, nb_mod_min
