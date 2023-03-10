import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import os
import webbrowser
import numpy as np

def generate_report(csv_file_path, report_path, report_title, params):
    df = pd.read_csv(csv_file_path)
    df = df.reset_index(drop=True)

    df_episode_reward = df.groupby("episode")["reward"].max()
    df_episode_modem = df.groupby("episode")["nb_modem_min"].min()
    df_episode_group = df.groupby("episode")["nb_group_min"].min()
    
    graph1 = px.line(df, x=df.index, y="reward", title="Reward vs. Timestep")
    graph2 = px.line(df, x=df.index, y="nb_modem_min", title="nb_modem_min vs. Timestep")
    graph3 = px.line(df, x=df.index, y="nb_group_min", title="nb_group_min vs. Timestep")
    graph4 = px.line(df_episode_reward, x=df_episode_reward.index, y="reward", title="Reward vs. Episode")
    graph5 = px.line(df_episode_modem, x=df_episode_modem.index, y="nb_modem_min", title="nb_modem_min vs. Episode")
    graph6 = px.line(df_episode_group, x=df_episode_group.index, y="nb_group_min", title="nb_group_min vs. Episode")

    with open(report_path, "w") as report_file:
        title = report_title.split("\n")
        report_file.write(f"<html><head><title>{report_title}</title></head><body>")
        report_file.write(f"<h1 align='center'>{title[0]}</h1>")
        report_file.write(f"<h2 align='center'>{title[1]}</h2>")

        report_file.write('<table style="border-collapse: collapse; width: 35%; margin: 0 auto; text-align: center;">')
        report_file.write('<tr><th style="border: 2px solid black; padding: 8px; font-weight: bold;">Experimental design</th><th style="border: 2px solid black; padding: 8px; font-weight: bold;;">Value</th></tr></thead>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of links</td><td style="border: 1px solid black; padding: 8px;">{params["Number of links"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of modems</td><td style="border: 1px solid black; padding: 8px;">{params["Number of modems"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of groups</td><td style="border: 1px solid black; padding: 8px;">{params["Number of groups"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Time</td><td style="border: 1px solid black; padding: 8px;">{params["Time"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of episodes</td><td style="border: 1px solid black; padding: 8px;">{params["Number of episodes"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of timesteps</td><td style="border: 1px solid black; padding: 8px;">{params["Number of timesteps"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of runs</td><td style="border: 1px solid black; padding: 8px;">{params["Number of runs"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Time out</td><td style="border: 1px solid black; padding: 8px;">{params["Time out"]}</td></tr></table>')

        report_file.write('<h2 align="center">Results</h2>')
        report_file.write('<h3 align="center">Episode</h3>')
        report_file.write(graph4.to_html(full_html=False, include_plotlyjs="cdn"))
        report_file.write(graph5.to_html(full_html=False, include_plotlyjs="cdn"))
        report_file.write(graph6.to_html(full_html=False, include_plotlyjs="cdn"))

        report_file.write('<h3 align="center">Timestep</h3>')
        report_file.write(graph1.to_html(full_html=False, include_plotlyjs="cdn"))
        report_file.write(graph2.to_html(full_html=False, include_plotlyjs="cdn"))
        report_file.write(graph3.to_html(full_html=False, include_plotlyjs="cdn"))
        
        report_file.write("</body></html>")
        
    print(f"Report saved to {os.path.abspath(report_path)}")
    webbrowser.open(os.path.abspath(report_path))

def create_graph_lines(dfs, y, title, names=None):
    if names is None:
        names = ["Run " + str(i+1) for i in range(len(dfs))]
    fig = go.Figure()
    for name, (i,df) in zip(names,enumerate(dfs)):
        if y is not None:
            fig.add_trace(go.Scatter(x=df.index, y=df[y], name=name, mode='lines'))
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df, name=name + str(i+1), mode='lines'))
    fig.update_layout(
        title=title,
        legend_title="Runs",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            traceorder="normal"
        )
    )
    return fig

def generate_report_runs(list_csv_file_path, report_path, report_title, params):
    dfs = [
        pd.read_csv(csv_file_path + "time_step_report.csv").reset_index(drop=True)
        for csv_file_path in list_csv_file_path
    ]

    df_episode_reward = [df.groupby("episode")["reward"].max() for df in dfs]
    df_episode_modem = [df.groupby("episode")["nb_modem_min"].min() for df in dfs]
    df_episode_group = [df.groupby("episode")["nb_group_min"].min() for df in dfs]
    graphs_reward_timestep = create_graph_lines(dfs, "reward", "Reward vs. Timestep")
    graphs_modem_timestep = create_graph_lines(dfs, "nb_modem_min", "nb_modem_min vs. Timestep")
    graphs_group_timestep = create_graph_lines(dfs, "nb_group_min", "nb_group_min vs. Timestep")
    graphs_reward_episode = create_graph_lines(df_episode_reward, None, "Reward vs. Episode")
    graphs_modem_episode = create_graph_lines(df_episode_modem, None, "nb_modem_min vs. Episode")
    graphs_group_episode = create_graph_lines(df_episode_group, None, "nb_group_min vs. Episode")


    with open(report_path, "w") as report_file:
        title = report_title.split("\n")
        report_file.write(f"<html><head><title>{report_title}</title></head><body>")
        report_file.write(f"<h1 align='center'>{title[0]}</h1>")
        report_file.write(f"<h2 align='center'>{title[1]}</h2>")

        report_file.write('<table style="border-collapse: collapse; width: 35%; margin: 0 auto; text-align: center;">')
        report_file.write('<tr><th style="border: 2px solid black; padding: 8px; font-weight: bold;">Experimental design</th><th style="border: 2px solid black; padding: 8px; font-weight: bold;;">Value</th></tr></thead>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of links</td><td style="border: 1px solid black; padding: 8px;">{params["Number of links"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of modems (min)</td><td style="border: 1px solid black; padding: 8px;">{params["Number of modems"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of groups (min)</td><td style="border: 1px solid black; padding: 8px;">{params["Number of groups"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Time</td><td style="border: 1px solid black; padding: 8px;">{params["Time"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of episodes</td><td style="border: 1px solid black; padding: 8px;">{params["Number of episodes"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of timesteps</td><td style="border: 1px solid black; padding: 8px;">{params["Number of timesteps"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of runs</td><td style="border: 1px solid black; padding: 8px;">{params["Number of runs"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Time out</td><td style="border: 1px solid black; padding: 8px;">{params["Time out"]}</td></tr></table>')

        report_file.write('<h2 align="center">Results</h2>')
        report_file.write('<h3 align="center">Episode</h3>')
        report_file.write(graphs_reward_episode.to_html(full_html=False, include_plotlyjs="cdn"))
        report_file.write(graphs_modem_episode.to_html(full_html=False, include_plotlyjs="cdn"))
        report_file.write(graphs_group_episode.to_html(full_html=False, include_plotlyjs="cdn"))

        report_file.write('<h3 align="center">Timestep</h3>')
        report_file.write(graphs_reward_timestep.to_html(full_html=False, include_plotlyjs="cdn"))
        report_file.write(graphs_modem_timestep.to_html(full_html=False, include_plotlyjs="cdn"))
        report_file.write(graphs_group_timestep.to_html(full_html=False, include_plotlyjs="cdn"))
        
        report_file.write("</body></html>")
        
    print(f"Report saved to {os.path.abspath(report_path)}")
    webbrowser.open(os.path.abspath(report_path))


def generate_report_comparison(list_csv_file_path_actor, list_csv_file_path_ppo, report_path, report_title, params):
    dfs_actor = [
        pd.read_csv(csv_file_path + "time_step_report.csv").reset_index(drop=True)
        for csv_file_path in list_csv_file_path_actor
    ]
    dfs_ppo = [
        pd.read_csv(csv_file_path + "time_step_report.csv").reset_index(drop=True)
        for csv_file_path in list_csv_file_path_ppo
    ]

    best_df_actor_index = np.argmin([df.apply(lambda x: x["nb_modem_min"] + x["nb_group_min"], axis=1).min() for df in dfs_actor])
    best_df_actor = dfs_actor[best_df_actor_index]
    best_df_ppo_index = np.argmin([df.apply(lambda x: x["nb_modem_min"] + x["nb_group_min"], axis=1).min() for df in dfs_ppo])
    best_df_ppo = dfs_ppo[best_df_ppo_index]
    worst_df_actor_index = np.argmax([df.apply(lambda x: x["nb_modem_min"] + x["nb_group_min"], axis=1).min() for df in dfs_actor])
    worst_df_actor = dfs_actor[worst_df_actor_index]
    worst_df_ppo_index = np.argmax([df.apply(lambda x: x["nb_modem_min"] + x["nb_group_min"], axis=1).min() for df in dfs_ppo])
    worst_df_ppo = dfs_ppo[worst_df_ppo_index]

    list_dfs = [best_df_actor, worst_df_actor, best_df_ppo, worst_df_ppo]
    names = ["Best Actor", "Worst Actor", "Best PPO", "Worst PPO"]

    df_episode_reward = [df.groupby("episode")["reward"].max() for df in list_dfs]
    df_episode_modem = [df.groupby("episode")["nb_modem_min"].min() for df in list_dfs]
    df_episode_group = [df.groupby("episode")["nb_group_min"].min() for df in list_dfs]
    graphs_reward_timestep = create_graph_lines(list_dfs, "reward", "Reward vs. Timestep", names)
    graphs_modem_timestep = create_graph_lines(list_dfs, "nb_modem_min", "nb_modem_min vs. Timestep", names)
    graphs_group_timestep = create_graph_lines(list_dfs, "nb_group_min", "nb_group_min vs. Timestep", names)

    graphs_reward_episode = create_graph_lines(df_episode_reward, None, "Reward vs. Episode")
    graphs_modem_episode = create_graph_lines(df_episode_modem, None, "nb_modem_min vs. Episode", names)
    graphs_group_episode = create_graph_lines(df_episode_group, None, "nb_group_min vs. Episode", names)

    with open(report_path, "w") as report_file:
        title = report_title.split("\n")
        report_file.write(f"<html><head><title>{report_title}</title></head><body>")
        report_file.write(f"<h1 align='center'>{title[0]}</h1>")
        report_file.write(f"<h2 align='center'>{title[1]}</h2>")
        report_file.write('<table style="border-collapse: collapse; width: 35%; margin: 0 auto; text-align: center;">')
        report_file.write('<tr><th style="border: 2px solid black; padding: 8px; font-weight: bold;">Experimental design</th><th style="border: 2px solid black; padding: 8px; font-weight: bold;;">Value</th></tr></thead>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of links</td><td style="border: 1px solid black; padding: 8px;">{params["Number of links"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of modems (min)</td><td style="border: 1px solid black; padding: 8px;">{params["Number of modems"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of groups (min)</td><td style="border: 1px solid black; padding: 8px;">{params["Number of groups"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Time</td><td style="border: 1px solid black; padding: 8px;">{params["Time"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of episodes</td><td style="border: 1px solid black; padding: 8px;">{params["Number of episodes"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of timesteps</td><td style="border: 1px solid black; padding: 8px;">{params["Number of timesteps"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Number of runs</td><td style="border: 1px solid black; padding: 8px;">{params["Number of runs"]}</td></tr>')
        report_file.write(f'<tr><td style="border: 1px solid black; padding: 8px;">Time out</td><td style="border: 1px solid black; padding: 8px;">{params["Time out"]}</td></tr></table>')

        report_file.write('<h2 align="center">Results</h2>')
        report_file.write('<h3 align="center">Episode</h3>')
        report_file.write(graphs_reward_episode.to_html(full_html=False, include_plotlyjs="cdn"))
        report_file.write(graphs_modem_episode.to_html(full_html=False, include_plotlyjs="cdn"))
        report_file.write(graphs_group_episode.to_html(full_html=False, include_plotlyjs="cdn"))

        report_file.write('<h3 align="center">Timestep</h3>')
        report_file.write(graphs_reward_timestep.to_html(full_html=False, include_plotlyjs="cdn"))
        report_file.write(graphs_modem_timestep.to_html(full_html=False, include_plotlyjs="cdn"))
        report_file.write(graphs_group_timestep.to_html(full_html=False, include_plotlyjs="cdn"))
        
        report_file.write("</body></html>")
        
    print(f"Report saved to {os.path.abspath(report_path)}")
    webbrowser.open(os.path.abspath(report_path))