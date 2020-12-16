# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
from flask_caching import Cache
import os
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import pickle
import json

external_stylesheets = [
    # Dash CSS
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    # Loading screen CSS
    'https://codepen.io/chriddyp/pen/brPBPO.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#app.config.suppress_callback_exceptions=True
'''CACHE_CONFIG = {
    # try 'filesystem' if you don't want to setup redis
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.environ.get('REDIS_URL', 'redis://localhost:6379')
}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)'''

N = 100

df_partidos = pd.read_csv('./dataset/csvs/partidos.csv')
df_materias = pd.read_csv('./dataset/csvs/materias.csv')
df_materias_comum = pd.read_csv('./dataset/csvs/materias_comum.csv')
df_parlamentares = pd.read_csv('./dataset/csvs/parlamentares.csv')

def clean_alt_list(list_):
    list_ = list_.replace(', ', '","')
    list_ = list_.replace('[', '["')
    list_ = list_.replace(']', '"]')
    return list_

def clean_alt_list_filiacoes(list_):
    list_ = list_.replace('}, ', '}","')
    list_ = list_.replace('[', '["')
    list_ = list_.replace(']', '"]')
    return list_

def limpa_colunas(df, coluna):
    if coluna != 'filiacoes':
        df[coluna] = df[coluna].apply(clean_alt_list)
    else:
        df[coluna] = df[coluna].apply(clean_alt_list_filiacoes)
    df[coluna] = df[coluna].apply(eval)
    return df

df_parlamentares = limpa_colunas(df_parlamentares, 'telefones')
df_parlamentares = limpa_colunas(df_parlamentares, 'filiacoes')
df_parlamentares = limpa_colunas(df_parlamentares, 'votos_favor')
df_parlamentares = limpa_colunas(df_parlamentares, 'votos_contra')
df_materias_comum = limpa_colunas(df_materias_comum, 'outros_autores')

tipos_rede = [{'label': 'Votos a Favor', 'value': 'favor'}, {'label': 'Votos Contra', 'value': 'contra'}]
filtros_senadores = [{'label': 'Somente Ativos', 'value': 'ativos'}, {'label': 'Somente Afastados', 'value': 'afastados'},\
                    {'label': 'Todos', 'value': 'todos'}, {'label': 'Partido', 'value': 'partido'},\
                    {'label': 'Alinhamento ao Governo', 'value': 'alinhamento'}, {'label': 'Customizar', 'value': 'customizar'}]
cores_nos = [{'label': 'Ativo/Afastado', 'value': 'ativo'}, {'label': 'Partido', 'value': 'partido'},\
            {'label': 'Alinhamento ao Governo', 'value': 'alinhamento'}, {'label': 'Número de Conexões', 'value': 'conexoes'}]

app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                id='tipo-rede',
                options=[{'label': tipos_rede[i]['label'], 'value': tipos_rede[i]['value']} for i in range(len(tipos_rede))],
                value=tipos_rede[0]['value'],
                placeholder='Tipo de Rede'
            ),
            dcc.Dropdown(
                id='filtro-senadores',
                options=[{'label': filtros_senadores[i]['label'], 'value': filtros_senadores[i]['value']} for i in range(len(filtros_senadores))],
                value=filtros_senadores[0]['value'],
                placeholder='Filtro de Parlamentares'
            ),
            dcc.Dropdown(
                id='coloracao-senadores',
                options=[{'label': cores_nos[i]['label'], 'value': cores_nos[i]['value']} for i in range(len(cores_nos))],
                value=cores_nos[0]['value'],
                placeholder='Coloração dos Nós'
            ),
            dcc.Checklist(
                id='mostrar-por-ano',
                options=[
                    {'label': 'Mostrar por Ano', 'value': 'ano'},
                ]
            ),
            dcc.Checklist(
                id='filtrar-senadores',
                options=[
                    {'label': 'Mostrar Somente Parlamentares com Conexão', 'value': 'filtrar'},
                ]
            ),
            html.Button('Gerar Grafo', id='botao-gerar-grafo', disabled=True,  n_clicks=0),
        ],
        style={'width': '49%', 'display': 'inline-block'}),
        
        html.Div(id='hidden-div-senadores', children=[
            dcc.Checklist(
                id='senadores-checklist',
                options=[{'label': df_parlamentares.iloc[i]['tratamento'] + df_parlamentares.iloc[i]['nome'],\
                'value': df_parlamentares.iloc[i]['cod']} for i in range(len(df_parlamentares))]
            )
        ],
        style={'width': '49%', 'display': 'inline-block', 'float': 'right', 'display': 'none', 'height': '100px', 'overflow-y': 'scroll'}),

        html.Div(id='hidden-div-partidos', children=[
            dcc.Checklist(
                id='partidos-checklist',
                options=[{'label': df_partidos.iloc[i]['nome'] + ' (' + df_partidos.iloc[i]['sigla'] + ')',\
                        'value': df_partidos.iloc[i]['cod']} for i in range(len(df_partidos))]
            )
        ],
        style={'width': '49%', 'display': 'inline-block', 'float': 'right', 'display': 'none', 'height': '100px', 'overflow-y': 'scroll'}),

        html.Div(id='hidden-div-alinhamento', children=[
            dcc.Checklist(
                id='alinhamentos-checklist',
                options=[{'label': df_partidos['alinhamento'].unique()[i], 'value': df_partidos['alinhamento'].unique()[i]} for i in range(len(df_partidos['alinhamento'].unique()))]
            )
        ],
        style={'width': '49%', 'display': 'inline-block', 'float': 'right', 'display': 'none', 'height': '100px', 'overflow-y': 'scroll'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    dcc.Graph(id='graph')
])

# perform expensive computations in this "global store"
# these computations are cached in a globally available
# redis memory store which is available across processes
# and for all time.
#@cache.memoize()
def global_store(value):
    if value == 'ativos':
        return df_parlamentares[df_parlamentares['afastado'] == 'Não']
    elif value == 'afastados':
        return df_parlamentares[df_parlamentares['afastado'] == 'Sim']
    elif value == 'todos' or value == 'customizar':
        return df_parlamentares

def filtra_senadores_custom(senadores):
    return df_parlamentares[df_parlamentares['cod'].isin(senadores)]

def filtra_senadores_partido(partidos):
    df_parlamentares = global_store('todos')
    index = len(df_parlamentares) - 1
    while index >= 0:
        if eval(df_parlamentares.iloc[index]['filiacoes'][0])['cod'] not in partidos:
            df_parlamentares = df_parlamentares.drop(df_parlamentares.index[index])
            index -= 1
        index -= 1
    return df_parlamentares

def filtra_senadores_alinhamento(alinhamentos):
    partidos = df_partidos[df_partidos['alinhamento'].isin(alinhamentos)]['cod'].tolist()
    return filtra_senadores_partido(partidos)

@app.callback(
    Output('hidden-div-senadores','style'),
    Input('filtro-senadores', 'value'))
def trigger_function1(filtro):
    if filtro == 'customizar':
        return {'width': '49%', 'display': 'inline-block', 'float': 'right', 'display': 'block', 'height': '100px', 'overflow-y': 'scroll'}
    else:
        return {'width': '49%', 'display': 'inline-block', 'float': 'right', 'display': 'none', 'height': '100px', 'overflow-y': 'scroll'}

@app.callback(
    Output('hidden-div-partidos','style'),
    Input('filtro-senadores', 'value'))
def trigger_function1(filtro):
    if filtro == 'partido':
        return {'width': '49%', 'display': 'inline-block', 'float': 'right', 'display': 'block', 'height': '100px', 'overflow-y': 'scroll'}
    else:
        return {'width': '49%', 'display': 'inline-block', 'float': 'right', 'display': 'none', 'height': '100px', 'overflow-y': 'scroll'}

@app.callback(
    Output('hidden-div-alinhamento','style'),
    Input('filtro-senadores', 'value'))
def trigger_function1(filtro):
    if filtro == 'alinhamento':
        return {'width': '49%', 'display': 'inline-block', 'float': 'right', 'display': 'block', 'height': '100px', 'overflow-y': 'scroll'}
    else:
        return {'width': '49%', 'display': 'inline-block', 'float': 'right', 'display': 'none', 'height': '100px', 'overflow-y': 'scroll'}

@app.callback(
    Output('botao-gerar-grafo','disabled'),
    Input('tipo-rede', 'value'),
    Input('filtro-senadores', 'value'),
    Input('coloracao-senadores', 'value'))
def trigger_function2(value_tipo, value_filtro, value_cor):
    if value_tipo is not None and value_filtro is not None and value_cor is not None:
        print("botão desbloqueado")
        return False
    print("botão bloqueado")
    return True

def extrai_listas(df_parlamentares, df_materias_comum):
    parlamentares = df_parlamentares['cod'].tolist()
    materias = df_materias_comum['cod'].tolist()
    return parlamentares, materias

def cria_mat_adj_votos(df_parlamentares, df_materias_comum, parlamentares, materias, tipo_voto):
    n = len(df_parlamentares)
    A = np.zeros((n,n))
    if tipo_voto != 'favor' and tipo_voto != 'contra':
        print("tipo de voto não existe")
        return
    if tipo_voto == 'favor':
        tipo_voto = 'votos_favor'
    if tipo_voto == 'contra':
        tipo_voto = 'votos_contra'
    for i in range(len(parlamentares)):
        votos = df_parlamentares.iloc[i][tipo_voto]
        for materia in votos:
            materia = materia.replace('\'', '')
            if materia == '':
                continue
            materia = int(materia)

            if materia not in materias:
                continue
            index_materia = materias.index(materia)
            autor_principal = int(df_materias_comum.iloc[index_materia]['autor_principal'])
            if autor_principal != 0:
                if autor_principal not in parlamentares:
                    continue
                index_parlamentar = parlamentares.index(autor_principal)
                A[i,index_parlamentar] += 1
            outros_autores = df_materias_comum.iloc[index_materia]['outros_autores']
            for autor in outros_autores:
                autor = autor.replace('\'', '')
                if autor == '':
                    continue
                autor = int(autor)
                if autor not in parlamentares:
                    continue
                index_parlamentar = parlamentares.index(autor)
                A[i,index_parlamentar] += 1
    return A

def filtra_mat_adj_votos(A, parlamentares):
    parlamentares_filtro = parlamentares.copy()
    counter = 0
    while counter < A.shape[0]:
        if np.all((A[counter,:counter] == 0)) and np.all((A[counter,counter+1:] == 0)) \
        and np.all((A[:counter,counter] == 0)) and np.all((A[counter+1:,counter] == 0)):
            A = np.delete(A, counter, 0)
            A = np.delete(A, counter, 1)
            del parlamentares_filtro[counter]
            counter = counter-1
        counter += 1
    return A, parlamentares_filtro

def filtra_df_parlamentares(df_parlamentares, parlamentares_filtro):
    df_parlamentares = df_parlamentares[df_parlamentares['cod'].isin(parlamentares_filtro)]
    return df_parlamentares

def colore_por_afastamento(df_parlamentares):
    colors_node = []
    for i in range(len(df_parlamentares)):
        if (df_parlamentares.afastado == "Sim").all():
            colors_node.append('darksalmon')
        else:
            colors_node.append('darkblue')
    return colors_node

def cria_trace(G, df_parlamentares, cores):
    pos = nx.random_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=cores,
            size=10,
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        if df_parlamentares.iloc[node].afastado == 'Sim':
            node_adjacencies.append(1)
        else:
            node_adjacencies.append(2)
        node_text.append(str(df_parlamentares.iloc[node].tratamento) + \
                        str(df_parlamentares.iloc[node].nome) )
        if pd.isnull(df_parlamentares.iloc[node].email) == False:
            node_text[-1] = node_text[-1] + '<br>email: ' + str(df_parlamentares.iloc[node].email)

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    return edge_trace, node_trace

def plotta_grafo(edge_trace, node_trace):
    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>Rede de Votações de Matérias de Parlamentares',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

@app.callback(
    Output('graph', 'figure'),
    Input('botao-gerar-grafo', 'n_clicks'),
    State('tipo-rede', 'value'),
    State('filtro-senadores', 'value'),
    State('coloracao-senadores', 'value'),
    State('filtrar-senadores', 'value'),
    State('senadores-checklist', 'value'),
    State('partidos-checklist', 'value'),
    State('alinhamentos-checklist', 'value'),
    State('mostrar-por-ano', 'value'))
def gera_nova_rede(n_cliques, tipo_rede, filtro_senadores, coloracao_nos, filtrar_senadores, senadores_checklist,\
                    partidos_checklist, alinhamentos_checklist, mostrar_por_ano):
    #if n_cliques == 0:
    #    return
    print("gerando rede")
    if filtro_senadores == 'customizar':
        df_parlamentares_filtro = filtra_senadores_custom(senadores_checklist)
    elif filtro_senadores == 'partido':
        df_parlamentares_filtro = filtra_senadores_partido(partidos_checklist)
    elif filtro_senadores == 'alinhamento':
        df_parlamentares_filtro = filtra_senadores_alinhamento(alinhamentos_checklist)
    else:
        df_parlamentares_filtro = global_store(filtro_senadores)
    parlamentares_filtro, materias = extrai_listas(df_parlamentares_filtro, df_materias_comum)
    A = cria_mat_adj_votos(df_parlamentares_filtro, df_materias_comum, parlamentares_filtro, materias, tipo_rede)
    if filtrar_senadores is not None and filtrar_senadores[0] == 'filtrar':
        A, parlamentares_filtro = filtra_mat_adj_votos(A, parlamentares_filtro)
        df_parlamentares_filtro = filtra_df_parlamentares(df_parlamentares_filtro, parlamentares_filtro)
    if coloracao_nos == 'ativo':
        cores_nos = colore_por_afastamento(df_parlamentares_filtro)
    G = nx.from_numpy_matrix(A)
    edge_trace, node_trace = cria_trace(G, df_parlamentares_filtro, cores_nos)
    fig = plotta_grafo(edge_trace, node_trace)
    fig.update_layout(transition_duration=500)
    print('update')
    return fig

@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('year-slider', 'value'))
def update_figure(selected_year):
    filtered_df = df[df.year == selected_year]

    fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp",
                     size="pop", color="continent", hover_name="country",
                     log_x=True, size_max=55)

    fig.update_layout(transition_duration=500)

    return fig


@app.callback(Output('signal', 'children'), Input('dropdown', 'value'))
def compute_value(value):
    # compute value and send a signal when done
    global_store(value)
    return value


@app.callback(Output('graph-1', 'figure'), Input('signal', 'children'))
def update_graph_1(value):
    # generate_figure gets data from `global_store`.
    # the data in `global_store` has already been computed
    # by the `compute_value` callback and the result is stored
    # in the global redis cached
    return generate_figure(value, {
        'data': [{
            'type': 'scatter',
            'mode': 'markers',
            'marker': {
                'opacity': 0.5,
                'size': 14,
                'line': {'border': 'thin darkgrey solid'}
            }
        }]
    })


@app.callback(Output('graph-2', 'figure'), Input('signal', 'children'))
def update_graph_2(value):
    return generate_figure(value, {
        'data': [{
            'type': 'scatter',
            'mode': 'lines',
            'line': {'shape': 'spline', 'width': 0.5},
        }]
    })


@app.callback(Output('graph-3', 'figure'), Input('signal', 'children'))
def update_graph_3(value):
    return generate_figure(value, {
        'data': [{
            'type': 'histogram2d',
        }]
    })


@app.callback(Output('graph-4', 'figure'), Input('signal', 'children'))
def update_graph_4(value):
    return generate_figure(value, {
        'data': [{
            'type': 'histogram2dcontour',
        }]
    })


if __name__ == '__main__':
    app.run_server(debug=True, threaded=True, processes=1)