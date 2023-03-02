from satellite_transmission.actor_critic import run_actor_critic
from satellite_transmission.ppo import run_ppo
from termcolor import colored
import numpy as np

def batch_comparison(links, algo, nb_episodes, duration_episode, nb_repeat, verbose):
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
    verbose = 0 if nb_repeat > 1 else 1

    for i in range(nb_repeat):
        print(f"Repeat {i+1}/{nb_repeat}...")
        if algo == "actor-critic":
            state_min, nb_grps_min, nb_mod_min = run_actor_critic(
                links, nb_episodes=nb_episodes, duration_episode=duration_episode, verbose=verbose
            )
            state_min_list.append(state_min)
            nb_grps_min_list.append(nb_grps_min)
            nb_mod_min_list.append(nb_mod_min)

        elif algo == "ppo":
            state_min, nb_grps_min, nb_mod_min = run_ppo(
                links, nb_episodes=nb_episodes, duration_episode=duration_episode, verbose=verbose
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
        print(colored("COMPARISON BETWEEN PPO AND ACTOR-CRITIC","blue"))
        print("=========================================")
        if best_ppo > best_actor:
            print(colored("PPO","green"),"is better most of the times", colored(f"{best_ppo = }","green"), colored(f"{best_actor = }","red"))
        else:
            print(colored("Actor-Critic","green"),"is better most of the times", colored(f"{best_actor = }","green"), colored(f"{best_ppo = }","red"))
        
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
            print(colored("PPO","green"), "found the best solution")
            state_min = state_min_ppo
            nb_grps_min = nb_grps_min_ppo
            nb_mod_min = nb_mod_min_ppo
        else:
            print(colored("Actor-Critic","green"), "found the best solution")
            state_min = state_min_actor
            nb_grps_min = nb_grps_min_actor
            nb_mod_min = nb_mod_min_actor
    else:
        best_index = np.argmin(nb_grps_min_list + nb_grps_min_list)
        state_min = state_min_list[best_index]
        nb_grps_min = nb_grps_min_list[best_index]
        nb_mod_min = nb_mod_min_list[best_index]

    return state_min, nb_grps_min, nb_mod_min