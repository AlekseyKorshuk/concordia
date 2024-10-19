from datetime import datetime
from datetime import timedelta
import json
import os
import time

import dash
from dash import callback_context
from dash import dash_table
from dash import dcc
from dash import html
from dash.dash_table.Format import Format
from dash.dash_table.Format import Scheme
from dash.dependencies import Input
from dash.dependencies import Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go


def parse_evaluation_dir(dir_name):
    parts = dir_name.split('__')
    if len(parts) >= 3:
        agent = parts[1]
        time_str = '__'.join(parts[2:])
        try:
            time = datetime.strptime(time_str, '%Y-%m-%d__%H:%M:%S')
            return agent, time
        except ValueError:
            return None, None
    return None, None

def process_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    return {
        'scenario': data.get('scenario'),
        'focal_per_capita_score': data.get('focal_per_capita_score'),
        'background_per_capita_score': data.get('background_per_capita_score'),
        'ungrouped_per_capita_score': data.get('ungrouped_per_capita_score'),
        'timestamp': datetime.fromtimestamp(os.path.getmtime(file_path))
    }

def load_data(directory='./evaluations'):
    results = {}
    scenarios = set()

    for eval_dir in os.listdir(directory):
        full_path = os.path.join(directory, eval_dir)
        if not os.path.isdir(full_path):
            continue

        agent, start_time = parse_evaluation_dir(eval_dir)
        if not agent or not start_time:
            continue

        agent_time = f"{agent}\n{start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        results[agent_time] = {'start_time': start_time, 'duration': 0}  # Initialize duration to 0
        end_time = start_time  # Initialize end_time

        json_files_found = False
        for json_file in os.listdir(full_path):
            if json_file.endswith('.json'):
                json_files_found = True
                json_data = process_json_file(os.path.join(full_path, json_file))
                if json_data['scenario']:
                    scenarios.add(json_data['scenario'])
                    results[agent_time][json_data['scenario']] = {
                        'focal': json_data['focal_per_capita_score'],
                        'background': json_data['background_per_capita_score'],
                        'ungrouped': json_data['ungrouped_per_capita_score']
                    }
                    end_time = max(end_time, json_data['timestamp'])  # Update end_time

        if json_files_found:
            # Calculate duration only if JSON files were found
            duration = (end_time - start_time).total_seconds() / 60  # Convert to minutes
            results[agent_time]['duration'] = round(duration, 2)

    return results, sorted(list(scenarios))

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

results, scenarios = load_data()

# Modify this part
df = pd.DataFrame(results).T.reset_index()
df = df.rename(columns={'index': 'Agent-Time'})
df['duration'] = df['duration'].fillna(0)  # Ensure duration exists and fill NaN with 0

# Melt the DataFrame
id_vars = ['Agent-Time', 'duration']
df_melted = df.melt(id_vars=id_vars, var_name='scenario', value_name='scores')

# Extract score types
df_melted[['focal', 'background', 'ungrouped']] = pd.json_normalize(df_melted['scores'])
df_melted = df_melted.drop('scores', axis=1)

# Pivot the melted DataFrame
df_pivot = df_melted.pivot(index='Agent-Time', columns='scenario', values=['focal', 'background', 'ungrouped'])
df_pivot.columns = [f'{col[1]}_{col[0]}' for col in df_pivot.columns]

# Add duration column
df_pivot['Duration'] = df_melted.groupby('Agent-Time')['duration'].first()

# Calculate overall scores
df_pivot['Overall_focal'] = df_pivot[[col for col in df_pivot.columns if col.endswith('_focal')]].sum(axis=1)
df_pivot['Overall_background'] = df_pivot[[col for col in df_pivot.columns if col.endswith('_background')]].sum(axis=1)
df_pivot['Overall_ungrouped'] = df_pivot[[col for col in df_pivot.columns if col.endswith('_ungrouped')]].sum(axis=1)

# Reorder columns to put Duration and Overall first
column_order = ['Duration', 'Overall_focal', 'Overall_background', 'Overall_ungrouped'] + [col for col in df_pivot.columns if not col.startswith('Overall') and col != 'Duration']
df_pivot = df_pivot[column_order].reset_index()

# Sort by Overall_focal in descending order
df_pivot = df_pivot.sort_values('Overall_focal', ascending=False)

app.layout = dbc.Container([
    html.H1('Evaluation Results', className='mt-4 mb-4 text-center'),
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Score Types"),
                dbc.CardBody(
                    dcc.Checklist(
                        id='score-type-selector',
                        options=[
                            {'label': 'Focal', 'value': 'focal'},
                            {'label': 'Background', 'value': 'background'},
                            {'label': 'Ungrouped', 'value': 'ungrouped'}
                        ],
                        value=['focal'],
                        inline=True,
                        className='mb-2'
                    )
                )
            ]),
            width=8
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Data Control"),
                dbc.CardBody(
                    dbc.Button("Reload Data", id="reload-button", color="primary", className="w-100")
                )
            ]),
            width=4
        ),
    ], className='mb-4'),
    dbc.Card([
        dbc.CardHeader("Results Table"),
        dbc.CardBody(
            dash_table.DataTable(
                id='results-table',
                sort_action='native',
                sort_mode='multi',
                sort_by=[{'column_id': 'Overall_focal', 'direction': 'desc'}],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'minWidth': '100px',
                    'maxWidth': '300px',
                    'whiteSpace': 'normal'
                },
                style_header={
                    'fontWeight': 'bold',
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'textAlign': 'center'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ],
                merge_duplicate_headers=True,
                page_size=10
            )
        )
    ]),
    dcc.Interval(
        id='interval-component',
        interval=5*60*1000,  # in milliseconds, set to 5 minutes
        n_intervals=0
    )
], fluid=True)

@app.callback(
    Output('results-table', 'columns'),
    Output('results-table', 'data'),
    Output('results-table', 'style_data_conditional'),
    Input('score-type-selector', 'value'),
    Input('reload-button', 'n_clicks'),
    Input('interval-component', 'n_intervals')
)
def update_table(selected_score_types, n_clicks, n_intervals):
    global results, scenarios, df, df_pivot

    # Check if the update was triggered by the reload button or interval
    if callback_context.triggered_id in ['reload-button', 'interval-component']:
        results, scenarios = load_data()
        df = pd.DataFrame(results).T.reset_index()
        df = df.rename(columns={'index': 'Agent-Time'})
        df['duration'] = df['duration'].fillna(0)  # Ensure duration exists and fill NaN with 0

        # Melt the DataFrame
        id_vars = ['Agent-Time', 'duration']
        df_melted = df.melt(id_vars=id_vars, var_name='scenario', value_name='scores')

        # Extract score types
        df_melted[['focal', 'background', 'ungrouped']] = pd.json_normalize(df_melted['scores'])
        df_melted = df_melted.drop('scores', axis=1)

        # Pivot the melted DataFrame
        df_pivot = df_melted.pivot(index='Agent-Time', columns='scenario', values=['focal', 'background', 'ungrouped'])
        df_pivot.columns = [f'{col[1]}_{col[0]}' for col in df_pivot.columns]

        # Add duration column
        df_pivot['Duration'] = df_melted.groupby('Agent-Time')['duration'].first()

        # Calculate overall scores
        df_pivot['Overall_focal'] = df_pivot[[col for col in df_pivot.columns if col.endswith('_focal')]].sum(axis=1)
        df_pivot['Overall_background'] = df_pivot[[col for col in df_pivot.columns if col.endswith('_background')]].sum(axis=1)
        df_pivot['Overall_ungrouped'] = df_pivot[[col for col in df_pivot.columns if col.endswith('_ungrouped')]].sum(axis=1)

        # Reorder columns to put Duration and Overall first
        column_order = ['Duration', 'Overall_focal', 'Overall_background', 'Overall_ungrouped'] + [col for col in df_pivot.columns if not col.startswith('Overall') and col != 'Duration']
        df_pivot = df_pivot[column_order].reset_index()

        # Sort by Overall_focal in descending order
        df_pivot = df_pivot.sort_values('Overall_focal', ascending=False)

    columns = [
        {'name': 'Agent-Time', 'id': 'Agent-Time'},
        {'name': 'Duration', 'id': 'Duration', 'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)}
    ]
    for scenario in ['Overall'] + scenarios:
        for score_type in selected_score_types:
            columns.append({
                'name': [scenario, score_type],
                'id': f'{scenario}_{score_type}',
                'type': 'numeric',
                'format': Format(precision=2, scheme=Scheme.fixed)
            })

    selected_columns = ['Agent-Time', 'Duration'] + [f'{scenario}_{score_type}' for scenario in ['Overall'] + scenarios for score_type in selected_score_types]
    data = df_pivot[selected_columns].to_dict('records')

    # Find the highest score for each column
    max_values = df_pivot[selected_columns[2:]].max()  # Start from index 2 to exclude Agent-Time and Duration

    # Create style_data_conditional to highlight the highest score in each column
    style_data_conditional = [
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        }
    ]

    for col in selected_columns[2:]:
        style_data_conditional.append({
            'if': {
                'filter_query': f'{{{col}}} = {max_values[col]}',
                'column_id': col
            },
            'backgroundColor': 'rgba(255, 255, 0, 0.3)',
            'fontWeight': 'bold'
        })

    return columns, data, style_data_conditional

if __name__ == '__main__':
    app.run_server(debug=True)
    app.run_server(debug=True)
