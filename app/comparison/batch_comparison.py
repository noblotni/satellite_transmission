import numpy as np
from termcolor import colored

from app.reinforcement_learning.agents.actor_critic import run_actor_critic
from app.reinforcement_learning.agents.ppo import run_ppo


def batch_comparison(links: list, algo: str, nb_episodes: int, duration_episode: int,
                     nb_repeat: int) -> tuple[int, int, int]:
    nb_grps_min_list: list[int] = []
    nb_mod_min_list: list[int] = []
    state_min_list: list[int] = []

    if algo == "compare":
        nb_grps_min_list_actor: list[int] = []
        nb_mod_min_list_actor: list[int] = []
        state_min_list_actor: list[int] = []
        nb_grps_min_list_ppo: list[int] = []
        nb_mod_min_list_ppo: list[int] = []
        state_min_list_ppo: list[int] = []

    best_ppo: int = 0
    best_actor: int = 0
    verbose: int = 0 if nb_repeat > 1 else 1

    for i in range(nb_repeat):
        print(f"Repeat {i + 1}/{nb_repeat}...")
        if algo == "actor-critic":
            state_min, nb_grps_min, nb_mod_min = run_actor_critic(
                links, nb_episodes=nb_episodes, duration_episode=duration_episode,
                verbose=verbose
            )
            state_min_list.append(state_min)
            nb_grps_min_list.append(nb_grps_min)
            nb_mod_min_list.append(nb_mod_min)

        elif algo == "ppo":
            state_min, nb_grps_min, nb_mod_min = run_ppo(
                links, nb_episodes=nb_episodes, duration_episode=duration_episode,
                verbose=verbose
            )
            state_min_list.append(state_min)
            nb_grps_min_list.append(nb_grps_min)
            nb_mod_min_list.append(nb_mod_min)

        elif algo == "compare":
            print('Actor-Critic...')
            state_min_actor, nb_grps_min_actor, nb_mod_min_actor = run_actor_critic(
                links, nb_episodes=nb_episodes, duration_episode=1000, verbose=verbose
            )
            print('PPO...')
            state_min_ppo, nb_grps_min_ppo, nb_mod_min_ppo = run_ppo(
                links, nb_episodes=nb_episodes, duration_episode=13000, verbose=verbose
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
            print(colored("PPO", "green"), "is better most of the times",
                  colored(f"{best_ppo = }", "green"),
                  colored(f"{best_actor = }", "red"))
        else:
            print(colored("Actor-Critic", "green"), "is better most of the times",
                  colored(f"{best_actor = }", "green"),
                  colored(f"{best_ppo = }", "red"))

        nb_grps_min_list_ppo: np.array = np.array(nb_grps_min_list_ppo)
        nb_mod_min_list_ppo: np.array = np.array(nb_mod_min_list_ppo)
        nb_grps_min_list_actor: np.array = np.array(nb_grps_min_list_actor)
        nb_mod_min_list_actor: np.array = np.array(nb_mod_min_list_actor)

        best_ppo_index: int = np.argmin(nb_grps_min_list_ppo + nb_mod_min_list_ppo)
        state_min_ppo: int = state_min_list_ppo[best_ppo_index]
        nb_grps_min_ppo: int = nb_grps_min_list_ppo[best_ppo_index]
        nb_mod_min_ppo: int = nb_mod_min_list_ppo[best_ppo_index]

        best_actor_index: int = np.argmin(
            nb_grps_min_list_actor + nb_mod_min_list_actor)
        state_min_actor: int = state_min_list_actor[best_actor_index]
        nb_grps_min_actor: int = nb_grps_min_list_actor[best_actor_index]
        nb_mod_min_actor: int = nb_mod_min_list_actor[best_actor_index]

        print(colored("Best PPO solution:", "blue"))
        print(f"nb_grps_min_ppo={colored(nb_grps_min_ppo, 'yellow')}")
        print(f"nb_mod_min_ppo={colored(nb_mod_min_ppo, 'yellow')}")

        print(colored("Best Actor-Critic solution:", "blue"))
        print(f"nb_grps_min_actor={colored(nb_grps_min_actor, 'yellow')}")
        print(f"nb_mod_min_actor={colored(nb_mod_min_actor, 'yellow')}")

        print("=========================================")
        if nb_grps_min_ppo + nb_mod_min_ppo < nb_grps_min_actor + nb_mod_min_actor:
            print(colored("PPO", "green"), "found the best solution")
            state_min: int = state_min_ppo
            nb_grps_min: int = nb_grps_min_ppo
            nb_mod_min: int = nb_mod_min_ppo
        else:
            print(colored("Actor-Critic", "green"), "found the best solution")
            state_min: int = state_min_actor
            nb_grps_min: int = nb_grps_min_actor
            nb_mod_min: int = nb_mod_min_actor
    else:
        best_index: int = np.argmin(nb_grps_min_list + nb_grps_min_list)
        state_min: int = state_min_list[best_index]
        nb_grps_min: int = nb_grps_min_list[best_index]
        nb_mod_min: int = nb_mod_min_list[best_index]

    return state_min, nb_grps_min, nb_mod_min
