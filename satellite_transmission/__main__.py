"""Run the optimization algorithm."""
import json
from pathlib import Path
import argparse
from satellite_transmission.actor_critic import run_actor_critic
from satellite_transmission.ppo import run_ppo
from satellite_transmission.json_report import generate_solution_report
import numpy as np
from termcolor import colored

def main(args):
    with open(args.links_path, "r") as file:
        links = json.load(file)

    nb_grps_min_list = []
    nb_mod_min_list = []
    state_min_list = []
    nb_grps_min_list_actor = []
    nb_mod_min_list_actor = []
    state_min_list_actor = []
    nb_grps_min_list_ppo = []
    nb_mod_min_list_ppo = []
    state_min_list_ppo = []
    best_ppo = 0
    best_actor = 0
    verbose = 0 if args.nb_repeat > 1 else 1

    print(f"Running {args.algo} algorithm...") if args.algo != "compare" else print("Running Actor-Critic and PPO algorithms...")
    print("=========================================")
    for i in range(args.nb_repeat):
        if args.nb_repeat > 1:
            print(f"Repeat {i+1}/{args.nb_repeat}...")
        if args.algo == "actor-critic":
            state_min, nb_grps_min, nb_mod_min = run_actor_critic(
                links, nb_episodes=args.nb_episodes, duration_episode=args.duration_episode, verbose=verbose
            )
            state_min_list.append(state_min)
            nb_grps_min_list.append(nb_grps_min)
            nb_mod_min_list.append(nb_mod_min)

        elif args.algo == "ppo":
            state_min, nb_grps_min, nb_mod_min = run_ppo(
                links, nb_episodes=args.nb_episodes, duration_episode=args.duration_episode, verbose=verbose
            )
            state_min_list.append(state_min)
            nb_grps_min_list.append(nb_grps_min)
            nb_mod_min_list.append(nb_mod_min)

        elif args.algo == "compare":
            print('Actor-Critic...')
            state_min_actor, nb_grps_min_actor, nb_mod_min_actor = run_actor_critic(
                links, nb_episodes=args.nb_episodes, duration_episode=1000, verbose=verbose
            )
            print('PPO...')
            state_min_ppo, nb_grps_min_ppo, nb_mod_min_ppo = run_ppo(
                links, nb_episodes=args.nb_episodes, duration_episode=13000, verbose=verbose
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
    if args.algo == "compare":
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

    print(colored("Best solution:","blue"))
    print(f"nb_grps_min={colored(nb_grps_min, 'yellow')}")
    print(f"nb_mod_min={colored(nb_mod_min, 'yellow')}")

    print("=========================================")
    print("Saving the solution...")
    generate_solution_report(
        state=state_min, links=links, output_path=args.output_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the optimization algorithm.")
    parser.add_argument("links_path", help="Path to the links data.", type=Path)
    parser.add_argument(
        "--algo",
        help="Optimization algorithm used. (default: actor-critic)",
        type=str,
        default="actor-critic",
    )
    parser.add_argument("--nb_episodes", help="Number of episodes.", type=int, default=10)
    parser.add_argument("--duration_episode", help="Duration of an episode.", type=int, default=10000)
    parser.add_argument("--output_path", "-o", help="Path to the output JSON file.", type=Path, default="./state_min.json")
    parser.add_argument("--nb_repeat", help="Number of times to repeat the optimization.", type=int, default=1)
    args = parser.parse_args()
    main(args)
