import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import glob
import numpy as np
import os


module_path = os.getcwd()
output_path = r"satellite_rl\output"
output_path = os.path.join(module_path, output_path)

all_actor_critic_runs_path = list(glob.glob(output_path + r"\actor-critic_Results\SatelliteRL\*"))
all_ppo_runs_path = list(glob.glob(output_path + r"\PPO_Results\SatelliteRL\*"))
all_comparison_runs_path = list(glob.glob(output_path + r"\comparison\SatelliteRL\*"))

all_runs_path = all_actor_critic_runs_path + all_ppo_runs_path + all_comparison_runs_path
options = []
for i,path in  enumerate(all_runs_path):
    options.append({
        "label": path.split("\\")[-1],
        "value": path
    })

all_runs_path_witout_comp = all_actor_critic_runs_path + all_ppo_runs_path
option_witout_comp = []
for i,path in  enumerate(all_runs_path_witout_comp):
    option_witout_comp.append({
        "label": path.split("\\")[-1],
        "value": path
    })

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H2("Report dashboard"), 
        html.Label('Select an option:', className="run-text-title"),
        dcc.RadioItems(
            id='option-selector',
            options=[
                {'label': 'Compare two runs', 'value': 'compare'},
                {'label': 'View a run', 'value': 'view'}
            ],
            value='view', className="run-text"
        ),
        html.Div([
            html.Label('Select a run:', className="run-text-title"),
            dcc.Dropdown(
                id='run-selector',
                options=options,
                value=options[0]["value"],
                searchable=False,
                clearable=False,
                className="run-dropdown"
            )
        ], id='run-selector-container'),
        html.Div([
            html.Label('Select run A:', className="run-text-title"),
            dcc.Dropdown(
                id='run-a-selector',
                options=option_witout_comp,
                value=option_witout_comp[0]["value"],
                searchable=False,
                clearable=False,
                className="run-dropdown"
            )
        ], id='run-a-selector-container', style={'display': 'none'}),
        html.Div([
            html.Label('Select run B:', className="run-text-title"),
            dcc.Dropdown(
                id='run-b-selector',
                options=option_witout_comp,
                value=option_witout_comp[1]["value"],
                searchable=False,
                clearable=False,
                className="run-dropdown"
            )
        ], id='run-b-selector-container', style={'display': 'none'}),
    ], className="column left"),

    html.Div([
        html.H1("Report dashboard"),
        html.Div([dash_table.DataTable(
            id='table',
            columns=[{'name': i, 'id': i} for i in [
                "Instance", "Number of episodes",
                "Number of timesteps",
                "Number of runs", "Time out",
                "Number of groups", "Number of modems",
                "Algorithm"
            ]],
            data=[],
            style_cell_conditional=[
                {
                    'if': {'column_id': 'Region'},
                    'textAlign': 'center'
                }
            ],
            style_cell={'padding': '5px', 'textAlign': 'center'},
            style_header={
                'backgroundColor': 'black',
                'fontWeight': 'bold',
                'color': 'white'
            },
            style_data={
                'whiteSpace': 'normal',
                'backgroundColor': 'white',
            }
        )]),
        html.Div([
            dcc.Graph(
                id='graph_boxplot',
                className="graph"
            )
        ], className="row", style={'display': 'none'}, id='graph_boxplot_container'),

        html.Div([
            dcc.Graph(
                id='graph_episode',
                className="graph"
            )
        ], className="row"),

        html.Div([
            dcc.Graph(
                id='graph_timestep',
                className="graph"
            )
        ], className="row")
    ], className="column right")
], className="container")

@app.callback(
    [Output('run-selector-container', 'style'),
     Output('run-a-selector-container', 'style'),
     Output('run-b-selector-container', 'style')],
    [Input('option-selector', 'value')]
)
def toggle_run_selector(option):
    if option == 'compare':
        return {'display': 'none'}, {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}

def load_metadata(path):
    metadata = pd.read_csv(os.path.join(path, "metadata.csv"), index_col=0)
    return metadata

def load_data(run, algo):
    if algo != "compare":
        dfs = [
            pd.read_csv(path).reset_index(drop=True)
            for path in glob.glob(f"{run}" + r"\report*.csv")
        ]
        return dfs
    else:
        dfs_actor = [ 
        pd.read_csv(path).reset_index(drop=True)
        for path in glob.glob(f"{run}" + r"\actor-critic\report*.csv")
        ]

        dfs_ppo = [
            pd.read_csv(path).reset_index(drop=True)
            for path in glob.glob(f"{run}" + r"\PPO\report*.csv")
        ]
        return dfs_actor, dfs_ppo

def get_episode_df(dfs):
    df_episode_reward = [df.groupby("episode")["reward"].max() for df in dfs]
    df_episode_modem = [df.groupby("episode")["nb_modem_min"].min() for df in dfs]
    df_episode_group = [df.groupby("episode")["nb_group_min"].min() for df in dfs]
    df_episodes = [pd.concat([df_ep_reward, df_ep_modem, df_ep_group], axis=1) for df_ep_reward, df_ep_modem, df_ep_group in zip(df_episode_reward, df_episode_modem, df_episode_group)]
    return df_episodes

def get_compare_df(dfs_actor, dfs_ppo):
    best_df_actor_index = np.argmin([df.apply(lambda x: x["nb_modem_min"] + x["nb_group_min"], axis=1).min() for df in dfs_actor])
    best_df_actor = dfs_actor[best_df_actor_index]
    best_df_ppo_index = np.argmin([df.apply(lambda x: x["nb_modem_min"] + x["nb_group_min"], axis=1).min() for df in dfs_ppo])
    best_df_ppo = dfs_ppo[best_df_ppo_index]
    worst_df_actor_index = np.argmax([df.apply(lambda x: x["nb_modem_min"] + x["nb_group_min"], axis=1).min() for df in dfs_actor])
    worst_df_actor = dfs_actor[worst_df_actor_index]
    worst_df_ppo_index = np.argmax([df.apply(lambda x: x["nb_modem_min"] + x["nb_group_min"], axis=1).min() for df in dfs_ppo])
    worst_df_ppo = dfs_ppo[worst_df_ppo_index]
    return best_df_actor, best_df_ppo, worst_df_actor, worst_df_ppo

def get_scatter_modem_group_reward(dfs, title, names=None):
    if names is None:
        names = range(1,len(dfs)+1)
    fig = go.Figure()
    for name, df in zip(names, dfs):
        fig.add_trace(go.Scatter(x=df.index, y=df["nb_modem_min"], name="nb_modem_min_" + str(name), mode='lines', yaxis="y1"))
        fig.add_trace(go.Scatter(x=df.index, y=df["nb_group_min"], name="nb_group_min_" + str(name), mode='lines', yaxis="y1"))
        fig.add_trace(go.Scatter(x=df.index, y=df["reward"], name="Reward_" + str(name), mode='lines', yaxis="y2"))
    fig.update_layout(
        title= {
                'text': title,
                'y':0.9, # new
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top' # new
                },
        xaxis=dict(title=title),
        yaxis=dict(title="Nombre de modem et de groupes"),
        yaxis2=dict(
            title="Reward",
            overlaying="y",
            side="right",
        ),
        legend = {
        "xanchor": "right", 
        "x": 1.4
        }
    )
    return fig

def get_box_modem_group_reward(dfs_best, title, names):
    fig = go.Figure()
    for name, y_axis in zip(names, ["y1", "y2", "y3", "y4"]):
        fig.add_trace(go.Box(y=dfs_best[name], name=name, yaxis=y_axis))
    fig.update_layout(
        title= {
                'text': title,
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
                },
        xaxis=dict(title="Timestep"),
        yaxis=dict(
            title="Nombre de modem et de groupes"
        ),
        yaxis2=dict(
            overlaying="y",
            anchor="free",
            autoshift=True,
            title="Modem"
        ),
        yaxis3=dict(
            overlaying="y",
            anchor="free",
            autoshift=True,
            title="Group"
        ),
        yaxis4=dict(
            overlaying="y",
            side="right",
            title="Reward"
        ),
        legend_title="Runs",
        legend = {
        "xanchor": "right", 
        "x": 1.4
        }
    )
    return fig


@app.callback(
    [Output('table', 'data'),
     Output('graph_boxplot_container', 'style'),
     Output('graph_episode', 'figure'),
     Output('graph_timestep', 'figure'),
     Output('graph_boxplot', 'figure')],
    [Input('run-selector', 'value'),
     Input('run-a-selector', 'value'),
     Input('run-b-selector', 'value')],
    [State('option-selector', 'value')]
)
def update_graphs(run, run_a, run_b, option):
    if option == 'compare':
        metadata_a = load_metadata(run_a).reset_index(drop=False)
        metadata_b = load_metadata(run_b).reset_index(drop=False)
        nb_runs_a = metadata_a['Number of runs'].iloc[0]
        nb_runs_b = metadata_b['Number of runs'].iloc[0]
        algo_a = metadata_a['Algorithm'].iloc[0]
        algo_b = metadata_b['Algorithm'].iloc[0]

        style_box = {'display': 'block'} if (nb_runs_a > 1) & (nb_runs_b > 1) else {'display': 'none'}

        metadata = pd.concat([metadata_a, metadata_b])
        table = metadata.to_dict('records')

        dfs_a = load_data(run_a,algo_a)
        dfs_b = load_data(run_b,algo_b)
        df_episodes_a = get_episode_df(dfs_a)
        df_episodes_b = get_episode_df(dfs_b)
        dfs = dfs_a + dfs_b
        df_episodes = df_episodes_a + df_episodes_b

        fig1 = get_scatter_modem_group_reward(df_episodes, "Episode", names=["algo_a_"+str(i) for i in range(1,nb_runs_a+1)] + ["algo_b_"+str(i) for i in range(1,nb_runs_b+1)])
        fig2 = get_scatter_modem_group_reward(dfs, "Timestep", names=["algo_a_"+str(i) for i in range(1,nb_runs_a+1)] + ["algo_b_"+str(i) for i in range(1,nb_runs_b+1)])

        if (nb_runs_a > 1) and (nb_runs_b > 1):
            dfs_modems_min_a = [df["nb_modem_min"].min() for df in dfs_a]
            dfs_groups_min_a = [df["nb_group_min"].min() for df in dfs_a]
            dfs_modems_min_b = [df["nb_modem_min"].min() for df in dfs_b]
            dfs_groups_min_b = [df["nb_group_min"].min() for df in dfs_b]
            
            
            dfs_best = pd.DataFrame({
                "Modem_a": dfs_modems_min_a, 
                "Modem_b": dfs_groups_min_a, 
                "Group_a": dfs_modems_min_b, 
                "Group_b": dfs_groups_min_b
            })
            fig3 = get_box_modem_group_reward(dfs_best, "Boxplot", ["Modem_a", "Modem_b", "Group_a", "Group_b"])
        else:
            fig3 = go.Figure()
    else:
        metadata = load_metadata(run).reset_index(drop=False)
        algo = metadata['Algorithm'].iloc[0]
        nb_runs = metadata['Number of runs'].iloc[0]
        style_box = {'display': 'block'} if (nb_runs > 1) else {'display': 'none'}
        table = metadata.to_dict('records')
        if algo != "compare":
            dfs = load_data(run,algo)
            df_episodes = get_episode_df(dfs)

            fig1 = get_scatter_modem_group_reward(df_episodes, "Episode")
            fig2 = get_scatter_modem_group_reward(dfs, "Timestep")

            dfs_add_min = [df.apply(lambda x: x["nb_modem_min"] + x["nb_group_min"], axis=1).min() for df in dfs]
            dfs_modems_min = [df["nb_modem_min"].min() for df in dfs]
            dfs_groups_min = [df["nb_group_min"].min() for df in dfs]
            dfs_reward_min = [df["reward"].min() for df in dfs]
            dfs_best = pd.DataFrame({
                "Modem_Group": dfs_add_min, 
                "Modem":dfs_modems_min, 
                "Group":dfs_groups_min, 
                "Reward":dfs_reward_min
            })
            fig3 = get_box_modem_group_reward(dfs_best, "Boxplot", ["Modem_Group", "Modem", "Group", "Reward"])

        else:
            dfs_actor, dfs_ppo = load_data(run,algo)
            best_df_actor, best_df_ppo, worst_df_actor, worst_df_ppo = get_compare_df(dfs_actor, dfs_ppo)
            best_df_actor_ep = get_episode_df([best_df_actor])[0]
            best_df_ppo_ep = get_episode_df([best_df_ppo])[0]
            worst_df_actor_ep = get_episode_df([worst_df_actor])[0]
            worst_df_ppo_ep = get_episode_df([worst_df_ppo])[0]
            dfs = [best_df_actor, best_df_ppo, worst_df_actor, worst_df_ppo]
            dfs_ep = [best_df_actor_ep, best_df_ppo_ep, worst_df_actor_ep, worst_df_ppo_ep]
            
            fig1 = get_scatter_modem_group_reward(dfs_ep, "Episode", names = ["Best Actor", "Best PPO", "Worst Actor", "Worst PPO"])
            fig2 = get_scatter_modem_group_reward(dfs, "Timestep", names = ["Best Actor", "Best PPO", "Worst Actor", "Worst PPO"])

            dfs_best = pd.DataFrame({
                "Modem_actor": [df["nb_modem_min"].min() for df in dfs_actor],
                "Group_actor": [df["nb_group_min"].min() for df in dfs_actor],
                "Modem_ppo": [df["nb_modem_min"].min() for df in dfs_ppo],
                "Group_ppo": [df["nb_group_min"].min() for df in dfs_ppo],
            })
            fig3 = get_box_modem_group_reward(dfs_best, "Boxplot Actor Critic", ["Modem_actor", "Group_actor", "Modem_ppo", "Group_ppo"])
        
    return table, style_box, fig1, fig2, fig3


if __name__ == '__main__':
    app.run_server(debug=True)