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
import json as js
from addEdge import addEdge

from analises import *

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

tipos_rede = [{'label': 'Votos a Favor', 'value': 'favor'},
              {'label': 'Votos Contra', 'value': 'contra'},
              {'label': 'Votos a Favor (Bipartido)', 'value': 'favor-bipartido'},
              {'label': 'Votos Contra (Bipartido)', 'value': 'contra-bipartido'}]
filtros_senadores = [{'label': 'Todos', 'value': 'todos'}, {'label': 'Somente Ativos', 'value': 'ativos'},
                     {'label': 'Somente Afastados', 'value': 'afastados'}, {'label': 'Partido', 'value': 'partido'},
                     {'label': 'Alinhamento ao Governo', 'value': 'alinhamento'}, {'label': 'Customizar', 'value': 'customizar'}]
cores_nos = [{'label': 'Ativo/Afastado', 'value': 'ativo'}, {'label': 'Partido', 'value': 'partido'},]
             #{'label': 'Alinhamento ao Governo', 'value': 'alinhamento'}, {'label': 'Número de Conexões', 'value': 'conexoes'}]
metricas_analise = [{'label': "Nenhuma",                        'value': 'analise-nenhuma'},
                    {'label': "Grau",                           'value': 'analise-grau'},
                    {'label': "Componentes Conexas",            'value': 'analise-connect'},
                    {'label': "Distância",                      'value': 'analise-dist'},
                    {'label': "Clusterização Local",            'value': 'analise-cluster'},
                    {'label': "Centralidade de Betweenness",    'value': 'analise-betwns'},
                    {'label': "Centralidade de Closeness",      'value': 'analise-closns'},
                    {'label': "Centralidade de PageRank",       'value': 'analise-pgrank'},
                    {'label': "Homofilia por Assortatividade",  'value': 'analise-assortatividade'},]
mostrar_tempo = [{'label': 'Nenhum', 'value':'tempo-todos'},
                 {'label': 'Mostrar por ano', 'value':'tempo-ano'},
                 {'label': 'Mostrar por mandato eleitoral', 'value':'tempo-mandato'}]
anos = [{'label':str(i), 'value':i} for i in range(1991,2021)]
mandatos_eleitorais = [{'label':'{} - {}'.format(i, i+3), 'value':i} for i in range(1991,2021,4)]
#[{'label': 'Todos', 'value':'mandato-todos'}]
cores_partidos = {
    'PDT' : px.colors.qualitative.Alphabet[0],
    'CIDADANIA' : px.colors.qualitative.Alphabet[1],
    'PODEMOS' : px.colors.qualitative.Alphabet[2],
    'PV': px.colors.qualitative.Alphabet[3],
    'PSD' : px.colors.qualitative.Alphabet[4],
    'PP' : px.colors.qualitative.Alphabet[5],
    'MDB' : px.colors.qualitative.Alphabet[6],
    'DEM' : px.colors.qualitative.Alphabet[7],
    'REDE' : px.colors.qualitative.Alphabet[8],
    'PROS' : px.colors.qualitative.Alphabet[9],
    'REPUBLICANOS' : px.colors.qualitative.Alphabet[10],
    'PT' : px.colors.qualitative.Alphabet[11],
    'PSDB' : px.colors.qualitative.Alphabet[12],
    'PL' : px.colors.qualitative.Alphabet[13],
    'PSB' : px.colors.qualitative.Alphabet[14],
    'PSL' : px.colors.qualitative.Alphabet[15],
    'PSC' : px.colors.qualitative.Alphabet[16],
    'PTB' : px.colors.qualitative.Alphabet[17],
    'SOLIDARIEDADE' : px.colors.qualitative.Alphabet[18],
    'PSOL' : px.colors.qualitative.Alphabet[19],
    'PMN' : px.colors.qualitative.Alphabet[20],
    'PHS' : px.colors.qualitative.Alphabet[21],
    'DC': px.colors.qualitative.Alphabet[22],
    'PSDC': px.colors.qualitative.Alphabet[23],
    'PPB': px.colors.qualitative.Alphabet[24],
    'PPR': px.colors.qualitative.Alphabet[25],
    'PMB': px.colors.qualitative.Set3[0],
    'PRTB': px.colors.qualitative.Set3[1],
    'PFL': px.colors.qualitative.Set3[2],
    'S/Partido' : 'black',
}

app.layout = html.Div([
    html.Div([
        html.H1("Redes do Senado Federal Brasileiro"),

        html.Div([
            dcc.Dropdown(
                id='tipo-rede',
                options=[{'label': tipos_rede[i]['label'], 'value': tipos_rede[i]['value']} for i in range(len(tipos_rede))],
                value=tipos_rede[0]['value'],   # NOTE: Comentando eu começo sem nenhum selecionado (e da pra ver o nome)
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
            dcc.Dropdown(
                id='filtrar-tempo',
                options=[{'label': mostrar_tempo[i]['label'], 'value': mostrar_tempo[i]['value']} for i in range(len(mostrar_tempo))],
                value=mostrar_tempo[0]['value'],
                placeholder='Filtro temporal',
                style={'margin-bottom': '7px'}
            ),
            dcc.Checklist(
                id='filtrar-senadores',
                options=[
                    {'label': 'Mostrar Somente Componentes com Conexão', 'value': 'filtrar'},
                ]
            ),
            dcc.Checklist(
                id='filtrar-materias',
                options=[
                    {'label': 'Filtrar Matérias por Subtipo', 'value': 'filtrar'}
                ]
            ),
            dcc.Checklist(
                id='fazer-analise',
                options=[
                    {'label': 'Caracterização por métricas', 'value': 'analise'},
                ],
                style={'margin-bottom': '4px'}
            ),
            html.Button('Gerar Grafo', id='botao-gerar-grafo', disabled=True,  n_clicks=0,
                        style={'margin-bottom': '6px'}),
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div(id='hidden-div-ano', children=[
            html.Div([
                dcc.Dropdown(
                    id='selecao-ano',
                    options=[{'label': anos[i]['label'], 'value': anos[i]['value']} for i in
                             range(len(anos))],
                    value=anos[-1]['value'],
                    placeholder='Ano',
                    searchable=True
                )
                # FIXME: Não estão aparecendo as opções!
            ],style={'width': '59%', 'display': 'inline-block', 'overflow-y': 'scroll'}),
        ],
        style={'width': '49%', 'float': 'right', 'display': 'none', 'height': '50px', 'backgroundColor': 'rgb(250, 250, 250)'}),

        html.Div(id='hidden-div-mandato', children=[
            html.Div([
                dcc.Dropdown(
                    id='selecao-mandato',
                    options=[{'label': mandatos_eleitorais[i]['label'], 'value': mandatos_eleitorais[i]['value']}
                             for i in range(len(mandatos_eleitorais))],
                    value=mandatos_eleitorais[-1]['value'],
                    placeholder='Mandato Eleitoral'
                )
            ],style={'width': '59%', 'height': '300px', 'display': 'inline-block', 'overflow-y': 'scroll'}),
        ],
        style={'width': '49%', 'float': 'right', 'display': 'none', 'height': '50px', 'backgroundColor': 'rgb(250, 250, 250)'}),
        
        html.Div(id='hidden-div-materias', children=[
            html.Div([
                dcc.Checklist(
                    id='materias-checklist',
                    options=[{'label': df_materias.nome_subtipo.unique()[i], 'value': df_materias.sigla_subtipo.unique()[i]} \
                        for i in range(len(df_materias.sigla_subtipo.unique()))]
                )
            ],
            style={'width': '59%', 'height': '200px', 'display': 'inline-block', 'overflow-y': 'scroll'}),
            html.Div([
                html.Button('Limpar', id='limpar-materias', disabled=True,  n_clicks=0),
            ])
        ],
        style={'width': '49%', 'float': 'right', 'display': 'none', 'height': '50px', 'backgroundColor': 'rgb(250, 250, 250)'}),

        html.Div(id='hidden-divs', children=[
            html.Div(id='div-filtros-senadores', children=[
                html.Div(id='hidden-div-senadores', children=[
                    html.Div([
                        dcc.Checklist(
                            id='senadores-checklist',
                            options=[
                                {'label': df_parlamentares.iloc[i]['tratamento'] + df_parlamentares.iloc[i]['nome'] + '\n',
                                'value': df_parlamentares.iloc[i]['cod']} for i in range(len(df_parlamentares))
                            ],
                            labelStyle={'display': 'block'}
                        )
                    ],
                    style={'height': '100px', 'display': 'inline-block', 'overflow-y': 'scroll'}),
                    html.Div([
                        html.Button('Limpar', id='limpar-senadores', disabled=True,  n_clicks=0),
                    ])
                ],
                style={'display': 'none'}),

                html.Div(id='hidden-div-partidos', children=[
                    html.Div([
                        dcc.Checklist(
                            id='partidos-checklist',
                            options=[{'label': df_partidos.iloc[i]['nome'] + ' (' + df_partidos.iloc[i]['sigla'] + ')',\
                                'value': df_partidos.iloc[i]['cod']} for i in range(len(df_partidos))]
                        )
                    ],
                    style={'height': '100px', 'display': 'inline-block', 'overflow-y': 'scroll'}),
                    html.Div([
                        html.Button('Limpar', id='limpar-partidos', disabled=True,  n_clicks=0),
                    ])
                ],
                style={'display': 'none'}),

                html.Div(id='hidden-div-alinhamentos', children=[
                    html.Div([
                        dcc.Checklist(
                            id='alinhamentos-checklist',
                            options=[{'label': df_partidos['alinhamento'].unique()[i], 'value': df_partidos['alinhamento'].unique()[i]} for i in range(len(df_partidos['alinhamento'].unique()))]
                        )
                    ],
                    style={'height': '100px', 'display': 'inline-block'}),
                    html.Div([
                        html.Button('Limpar', id='limpar-alinhamentos', disabled=True,  n_clicks=0),
                    ])
                ],
                style={'display': 'none'})
            ],
            style={'float': 'top', 'height': '49%'}),

            html.Div(id='hidden-div-analises', children=[
                html.Div([
                    dcc.Dropdown(
                        id='selecao-analise',
                        options=[{'label': metricas_analise[i]['label'], 'value': metricas_analise[i]['value']} for i in
                                 range(len(metricas_analise))],
                        value=metricas_analise[0]['value'],
                        placeholder='Métrica Analisada'
                    )
                ],
                style={'display': 'inline-block', 'float': 'left', 'width': '79%', 'height': '99%'}),
            ],
            style={'display': 'none', 'float': 'bottom', 'height': '200px', 'backgroundColor': 'rgb(250, 250, 250)'}),
        ],
        style={'width': '49%', 'float': 'right'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    dcc.Graph(id='graph'),

    html.Div([
        html.Div(id='info-metricas', children=[
            html.H3("Análises da rede"),    # NOTE: trocar o título de acordo com a analise?

            html.Div(id='detalhes-analises'),
        ], style={'width': '99%', 'float': 'center', 'display': 'none', 'height': '400px'})
    ], style={
        'borderTop': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    })
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

def global_store_materias():
    return df_materias

def global_store_materias_comum():
    return df_materias_comum

def filtra_senadores_custom(senadores):
    return df_parlamentares[df_parlamentares['cod'].isin(senadores)]

def filtra_materias_subtipo(df_materias, materias):
    df_materias = df_materias[df_materias['sigla_subtipo'].isin(materias)]
    return df_materias

def filtra_senadores_partido(partidos):
    df_parlamentares = global_store('todos')
    index = len(df_parlamentares) - 1
    while index >= 0:
        if int(eval(df_parlamentares.iloc[index]['filiacoes'][0])['cod']) not in partidos:
            df_parlamentares = df_parlamentares.drop(df_parlamentares.index[index])
        index -= 1
    return df_parlamentares

def filtra_senadores_alinhamento(alinhamentos):
    partidos = df_partidos[df_partidos['alinhamento'].isin(alinhamentos)]['cod'].tolist()
    return filtra_senadores_partido(partidos)

@app.callback(
    Output('hidden-div-senadores','style'),
    Input('filtro-senadores', 'value'))
def toggle_custom_senadores(filtro):
    if filtro == 'customizar':
        return {'display': 'inline-block'}
    else:
        return {'display': 'none'}

@app.callback(
    Output('hidden-div-partidos','style'),
    Input('filtro-senadores', 'value'))
def toggle_custom_partidos(filtro):
    if filtro == 'partido':
        return {'display': 'inline-block'}
    else:
        return {'display': 'none'}

@app.callback(
    Output('hidden-div-alinhamentos','style'),
    Input('filtro-senadores', 'value'))
def toggle_custom_alinhamentos(filtro):
    if filtro == 'alinhamento':
        return {'display': 'inline-block'}
    else:
        return {'display': 'none'}

@app.callback(
    [Output('hidden-div-analises','style'),
     Output('selecao-analise', 'value')],
    [Input('fazer-analise', 'value')])
def toggle_select_metric(filtro):
    # FIXME: Quando coloco customização de senadores ao mesmo tempo que selecao de metrica o layout fica estranho
    if filtro is not None and len(filtro) > 0 and filtro[0] == 'analise':
        return {'display': 'block', 'float': 'left', 'height': '200px', 'width': '59%'}, \
               'analise-nenhuma'
    else:
        return {'display': 'none', 'float': 'left', 'height': '200px'}, 'analise-nenhuma'

@app.callback(
    Output('hidden-div-materias','style'),
    Input('filtrar-materias', 'value'))
def toggle_custom_materias(filtro):
    if filtro is not None and len(filtro) > 0:
        return {'width': '49%', 'display': 'inline-block', 'float': 'right', 'display': 'block', 'height': '10px'}
    else:
        return {'width': '49%', 'float': 'right', 'display': 'none'}

@app.callback(
    [Output('hidden-div-ano','style'),
     Output('hidden-div-mandato','style')],
    [Input('filtrar-tempo', 'value')]
)
def toggle_custom_time(filtro):
    if filtro == 'tempo-ano':
        return {'width': '49%', 'display': 'inline-block', 'float': 'right', 'display': 'block', 'height': '10px'}, \
               {'width': '49%', 'float': 'right', 'display': 'none'}
    elif filtro == 'tempo-mandato':
        return {'width': '49%', 'float': 'right', 'display': 'none'}, \
               {'width': '49%', 'display': 'inline-block', 'float': 'right', 'display': 'block', 'height': '10px'}
    else:
        return {'width': '49%', 'float': 'right', 'display': 'none'}, {'width': '49%', 'float': 'right', 'display': 'none'}

# FIXME: Layout esquisito quanto se mistura hidden divs!
#   Não consegui resolver diminuindo o tamanho dos divs....


@app.callback(
    Output('info-metricas','style'),
    Input('selecao-analise', 'value'))
def show_analysis(filtro):
    if filtro == 'analise-nenhuma':
        return {'width': '99%', 'float': 'center', 'display': 'none', 'height': '400px', 'backgroundColor': 'rgb(250, 250, 250)'}
    else:
        return {'width': '90%', 'float': 'center', 'display': 'block', 'height': '400px', 'backgroundColor': 'rgb(250, 250, 250)'}

@app.callback(
    Output('botao-gerar-grafo','disabled'),
    Input('tipo-rede', 'value'),
    Input('filtro-senadores', 'value'),
    Input('coloracao-senadores', 'value'))
def toggle_botao_gerar(value_tipo, value_filtro, value_cor):
    if value_tipo is not None and value_filtro is not None and value_cor is not None:
        print("botão desbloqueado")
        return False
    print("botão bloqueado")
    return True

@app.callback(
    Output('limpar-senadores','disabled'),
    Input('senadores-checklist', 'value'))
def toggle_botao_limpar_senadores(checklist):
    if checklist is not None and len(checklist) > 0:
        return False
    return True

@app.callback(
    Output('limpar-materias','disabled'),
    Input('materias-checklist', 'value'))
def toggle_botao_limpar_materias(checklist):
    if checklist is not None and len(checklist) > 0:
        return False
    return True

@app.callback(
    Output('limpar-partidos','disabled'),
    Input('partidos-checklist', 'value'))
def toggle_botao_limpar_partidos(checklist):
    if checklist is not None and len(checklist) > 0:
        return False
    return True

@app.callback(
    Output('limpar-alinhamentos','disabled'),
    Input('alinhamentos-checklist', 'value'))
def toggle_botao_limpar_alinhamentos(checklist):
    if checklist is not None and len(checklist) > 0:
        return False
    return True

@app.callback(
    Output('senadores-checklist','value'),
    Input('limpar-senadores', 'n_clicks'))
def limpa_senadores(n_cliques):
    print('clicando botão')
    return []

@app.callback(
    Output('partidos-checklist','value'),
    Input('limpar-partidos', 'n_clicks'))
def limpa_partidos(n_cliques):
    return []

@app.callback(
    Output('alinhamentos-checklist','value'),
    Input('limpar-alinhamentos', 'n_clicks'))
def limpa_alinhamentos(n_cliques):
    return []

def cria_mat_adj_votos(df_parlamentares, df_materias_comum, tipo_voto):
    parlamentares = df_parlamentares['cod'].tolist()
    materias = df_materias_comum['cod'].tolist()
    n = len(df_parlamentares)
    A = np.zeros((n,n))
    if tipo_voto == 'favor':
        tipo_voto = 'votos_favor'
    elif tipo_voto == 'contra':
        tipo_voto = 'votos_contra'
    else:
        print("tipo de voto não existe")
        return
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

def filtra_mat_adj_votos(A, df_parlamentares):
    df_parlamentares_filtro = df_parlamentares.copy()
    counter = 0
    while counter < A.shape[0]:
        if np.all(A[counter,:counter] == 0) and np.all(A[counter,counter+1:] == 0) \
        and np.all(A[:counter,counter] == 0) and np.all(A[counter+1:,counter] == 0):
            A = np.delete(A, counter, 0)
            A = np.delete(A, counter, 1)
            df_parlamentares_filtro = df_parlamentares_filtro.drop(df_parlamentares_filtro.index[counter], axis=0)
            counter = counter-1
        counter += 1
    return A, df_parlamentares_filtro

def filtra_mat_adj_bipartido(A, df_parlamentares, df_materias):
    df_parlamentares_filtro = df_parlamentares.copy()
    df_materias_filtro = df_materias.copy()
    counter = 0
    print('tam mat antes', A.shape[0])
    while counter < A.shape[0]:
        if np.all(A[counter,:counter] == 0) and np.all(A[counter,counter+1:] == 0) \
        and np.all(A[:counter,counter] == 0) and np.all(A[counter+1:,counter] == 0):
            A = np.delete(A, counter, 0)
            A = np.delete(A, counter, 1)
            if counter >= len(df_parlamentares_filtro):
                df_materias_filtro = df_materias_filtro.drop(df_materias_filtro.index[counter%len(df_parlamentares_filtro)], axis=0)
            else:
                df_parlamentares_filtro = df_parlamentares_filtro.drop(df_parlamentares_filtro.index[counter], axis=0)
            counter = counter-1
        counter += 1
    print('tam mat depois', A.shape[0])
    return A, df_parlamentares_filtro, df_materias_filtro

def filtra_dfs_bipartido(df_parlamentares, parlamentares_filtro, df_materias, materias_filtro):
    df_parlamentares = df_parlamentares[df_parlamentares['cod'].isin(parlamentares_filtro)]
    df_materias = df_materias[df_materias['cod'].isin(materias_filtro)]
    return df_parlamentares, df_materias

def colore_por_afastamento(df_parlamentares):
    colors_node = []
    colors_dict = {}
    for i in range(len(df_parlamentares)):
        if df_parlamentares.iloc[i].afastado == 'Sim': #(df_parlamentares.afastado == "Sim").all():
            colors_node.append('red')       # darksalmon
            colors_dict[i] = "Afastado"
        else:
            colors_node.append('green')     # darkblue
            colors_dict[i] = "Ativo"
    return colors_node, colors_dict

def colore_por_partido(tab_parlamentares, ano):
    colors_node = []
    colors_dict = {}

    for i in range(len(tab_parlamentares)):
        partido = get_curr_partido(tab_parlamentares.iloc[i].filiacoes, ano)
        colors_node.append(cores_partidos[partido])
        colors_dict[i] = partido

    return colors_node, colors_dict

def get_curr_partido(filiacoes, ano):
    codigo = None

    for filiacao in filiacoes:
        jsstr = filiacao.replace("\'", "\"")
        jsstr = jsstr.replace("None", "\"None\"")

        fil = js.loads(jsstr)

        if int(fil['data_filiacao'][:4]) <= ano:
            termino = fil['data_desfiliacao']
            if termino == 'None':
                # Achei o partido!
                codigo = int(fil['cod'])
                break

            if int(termino[:4]) > ano:
                codigo = int(fil['cod'])
                break

    if not codigo:
        return 'S/Partido'

    partido = df_partidos[df_partidos['cod'] == codigo]
    partido = partido['sigla'].iloc[0]

    return partido


def colore_por_atributo(df_parlamentares, coloracao_nos):
    colors_node = []
    colors_dict = {}
    if coloracao_nos == 'ativo':
        for i in range(len(df_parlamentares)):
            if df_parlamentares.iloc[i].afastado == 'Sim': #(df_parlamentares.afastado == "Sim").all():
                colors_node.append('red')       # darksalmon
            else:
                colors_node.append('green')     # darkblue
    if coloracao_nos == 'partido':
        return colore_por_partido(df_parlamentares, )
    return colors_node

'''def cria_trace(G, df_parlamentares, cores):
    pos = nx.circular_layout(G) #nx.random_layout(G)
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
    # TODO: Acho que para colocar legenda o ideal é a gente separar em dois traces...
    # Ou adicionar como atributo de alguma forma e colocar agrupado assim
    # https://plotly.com/python/legend/

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
            node_text[-1] = node_text[-1] + \
                            '<br>e-mail: ' + str(df_parlamentares.iloc[node].email)# + \
                            #'<br>UF: ' + str(df_parlamentares.iloc[node].uf)
            # FIXME: O csv dos parlamentares nao tem partido!!!

    #node_trace.marker.color = node_adjacencies
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
    return fig'''

def cria_grafo_bipartido(df_parlamentares, df_materias, tipo_voto):
    parlamentares = df_parlamentares['cod'].tolist()
    materias = df_materias['cod'].tolist()
    if tipo_voto == 'favor-bipartido':
        tipo_voto = 'votos_favor'
    elif tipo_voto == 'contra-bipartido':
        tipo_voto = 'votos_contra'
    else:
        print("tipo de voto não existe")
        return
    n = len(df_parlamentares)
    m = len(df_materias)
    A = np.zeros((n,m))
    G = nx.Graph()
    G.add_nodes_from(parlamentares, bipartite=0)
    G.add_nodes_from(materias, bipartite=1)
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
            G.add_edges_from([(parlamentares[i], materias[index_materia])])
    return G


def filtrar_por_ano(df_info, ano):
    return df_info.loc[df_info['ano_votacao'] == ano]


def filtrar_por_mandato(df_info, ano):
    return df_info.loc[(df_info['ano_votacao'] >= ano) & (df_info['ano_votacao'] <= ano+3)]


def analisar_grafo(graph, metrica, atributo=None):
    if metrica == 'analise-nenhuma':
        return []

    elif metrica == 'analise-grau':
        print("DEBUG:\tAnálise de grau")    # TODO: adicionar log! (com 'import logging')
        gmin, gmax, gmed, gdp, gmediana, gdistr = degree_analysis(graph)

        children = []

        textual = [html.H4("Grau dos nós"),
                   html.P("Mínimo: {}".format(gmin)),
                   html.P("Máximo: {}".format(gmax)),
                   html.P("Mediana: {}".format(gmediana)),
                   html.P("Média: {:.2f}".format(gmed)),
                   html.P("Desvio Padrão: {:.2f}".format(gdp)),
                  ]
        children += [html.Div(textual, style={'width':'40%', 'float':'left', 'display':'inline-block'})]

        fig = go.Figure(
            data=[go.Scatter(x=gdistr[0], y=gdistr[1], mode="lines+markers",
                             marker_color="rgba(0, 0, 0, .7)", line_color="rgba(0, 0, 255, 0.6)")],
            layout=go.Layout(
                title=go.layout.Title(text="Função Distribuição de Probabilidade"),
                height=350
            )
        )
        children += [html.Div([dcc.Graph(figure=fig)], style={'width':'49%', 'display': 'inline-block'})]

        return children

    elif metrica == 'analise-cluster':
        clmin, clmax, clmed, cldp, clmediana, cldistr = clustering_analysis(graph)
        print("DEBUG:\tAnálise de clusterização")

        children = []

        textual = [html.H4("Clusterização local dos nós"),
                   html.P("Mínimo: {:.3f}".format(clmin)),
                   html.P("Máximo: {:.3f}".format(clmax)),
                   html.P("Mediana: {:.3f}".format(clmediana)),
                   html.P("Média: {:.3f}".format(clmed)),
                   html.P("Desvio Padrão: {:.3f}".format(cldp)),
                   ]
        children += [html.Div(textual, style={'width': '40%', 'float': 'left', 'display': 'inline-block'})]

        fig = go.Figure(
            data=[go.Scatter(x=cldistr[0], y=cldistr[1], mode="lines")],
            layout=go.Layout(
                title=go.layout.Title(text="Função de Distribuição Cumulativa Complementar"),
                height=350
            )
        )
        children += [html.Div([dcc.Graph(figure=fig)], style={'width': '49%', 'display': 'inline-block'})]

        return children

    elif metrica == "analise-connect":
        cn_n, cnmin, cnmax, cnmed, cndp, cnmediana, cndistr = connexity_analysis(graph)
        print("DEBUG:\tAnálise das componentes conexas")

        children = []

        textual = [html.H4("Tamanhos das componentes conexas"),
                   html.P("Mínimo: {}".format(cnmin)),
                   html.P("Máximo: {}".format(cnmax)),
                   html.P("Mediana: {}".format(cnmediana)),
                   html.P("Média: {:.2f}".format(cnmed)),
                   html.P("Desvio Padrão: {:.2f}".format(cndp)),
                   html.H4("Total de {} componentes".format(cn_n)),
                   ]
        children += [html.Div(textual, style={'width': '40%', 'float': 'left', 'display': 'inline-block'})]

        fig = go.Figure(
            data=[go.Scatter(x=cndistr[0], y=cndistr[1], mode="lines+markers",
                             marker_color="rgba(0, 0, 0, .7)", line_color="rgba(0, 0, 255, 0.6)")],
            layout=go.Layout(
                title=go.layout.Title(text="Função Distribuição de Probabilidade"),
                height=350
            )
        )
        children += [html.Div([dcc.Graph(figure=fig)], style={'width': '49%', 'display': 'inline-block'})]

        return children

    elif metrica == "analise-dist":
        dmin, dmax, dmed, ddp, dmediana, ddistr, d_n = distance_analysis(graph)
        print("DEBUG:\tAnálise de distância")

        children = []

        textual = [html.H4("Distâncias entre os nós"),
                   html.P("Mínimo: {}".format(dmin)),
                   html.P("Máximo: {}".format(dmax)),
                   html.P("Mediana: {}".format(dmediana)),
                   html.P("Média: {:.2f}".format(dmed)),
                   html.P("Desvio Padrão: {:.2f}".format(ddp)),
                   html.H4("Total de {} pares de vértices".format(d_n)),
                   html.P("Observação: são considerados apenas os pares de vértices conectados entre si"),
                   ]
        children += [html.Div(textual, style={'width': '40%', 'float': 'left', 'display': 'inline-block'})]

        fig = go.Figure(
            data=[go.Scatter(x=ddistr[0], y=ddistr[1], mode="lines+markers",
                             marker_color="rgba(0, 0, 0, .7)", line_color="rgba(0, 0, 255, 0.6)")],
            layout=go.Layout(
                title=go.layout.Title(text="Função Distribuição de Probabilidade"),
                height=350
            )
        )
        children += [html.Div([dcc.Graph(figure=fig)], style={'width': '49%', 'display': 'inline-block'})]

        return children

    elif metrica == "analise-betwns":
        btwnmin, btwnmax, btwnmed, btwndp, btwnmediana, btwndistr = betweenness_analysis(graph)
        print("DEBUG:\tAnálise de betweenness")

        children = []

        textual = [html.H4("Centralidade dos nós por Betweenness"),
                   html.P("Mínimo: {:.4f}".format(btwnmin)),
                   html.P("Máximo: {:.4f}".format(btwnmax)),
                   html.P("Mediana: {:.4f}".format(btwnmediana)),
                   html.P("Média: {:.4f}".format(btwnmed)),
                   html.P("Desvio Padrão: {:.4f}".format(btwndp)),
                   ]
        children += [html.Div(textual, style={'width': '40%', 'float': 'left', 'display': 'inline-block'})]

        fig = go.Figure(
            data=[go.Scatter(x=btwndistr[0], y=btwndistr[1], mode="lines")],
            layout=go.Layout(
                title=go.layout.Title(text="Função de Distribuição Cumulativa Complementar"),
                height=350
            )
        )
        children += [html.Div([dcc.Graph(figure=fig)], style={'width': '49%', 'display': 'inline-block'})]

        return children

    elif metrica == "analise-closns":
        closemin, closemax, closemed, closedp, closemediana, closedistr = closeness_analysis(graph)
        print("DEBUG:\tAnálise de closeness")

        children = []

        textual = [html.H4("Centralidade dos nós por Closeness"),
                   html.P("Mínimo: {:.3f}".format(closemin)),
                   html.P("Máximo: {:.3f}".format(closemax)),
                   html.P("Mediana: {:.3f}".format(closemediana)),
                   html.P("Média: {:.3f}".format(closemed)),
                   html.P("Desvio Padrão: {:.3f}".format(closedp)),
                   ]
        children += [html.Div(textual, style={'width': '40%', 'float': 'left', 'display': 'inline-block'})]

        fig = go.Figure(
            data=[go.Scatter(x=closedistr[0], y=closedistr[1], mode="lines")],
            layout=go.Layout(
                title=go.layout.Title(text="Função de Distribuição Cumulativa Complementar"),
                height=350
            )
        )
        children += [html.Div([dcc.Graph(figure=fig)], style={'width': '49%', 'display': 'inline-block'})]

        return children

    elif metrica == "analise-pgrank":
        pgrkmin, pgrkmax, pgrkmed, pgrkdp, pgrkmediana, pgrkdistr = pagerank_analysis(graph)
        print("DEBUG:\tAnálise de PageRank (alfa=0.85)")
        # TODO: Permitir mudança do alfa

        children = []

        textual = [html.H4("Centralidade dos nós por PageRank (0,85)"),
                   html.P("Mínimo: {:.4f}".format(pgrkmin)),
                   html.P("Máximo: {:.4f}".format(pgrkmax)),
                   html.P("Mediana: {:.4f}".format(pgrkmediana)),
                   html.P("Média: {:.4f}".format(pgrkmed)),
                   html.P("Desvio Padrão: {:.4f}".format(pgrkdp)),
                   ]
        children += [html.Div(textual, style={'width': '40%', 'float': 'left', 'display': 'inline-block'})]

        fig = go.Figure(
            data=[go.Scatter(x=pgrkdistr[0], y=pgrkdistr[1], mode="lines")],
            layout=go.Layout(
                title=go.layout.Title(text="Função de Distribuição Cumulativa Complementar"),
                height=350
            )
        )
        children += [html.Div([dcc.Graph(figure=fig)], style={'width': '49%', 'display': 'inline-block'})]

        return children

    elif metrica == "analise-assortatividade":
        print("DEBUG:\tAnálise de Homofilia por assortatividade")

        # TODO: Colocar essas variáveis como constantes externas à função? Ai nao preciso ficar recriando
        mapeamento = {"Ativo/Afastado": {"Ativo": 0, "Afastado": 1},}
        descricao = {"Ativo/Afastado": "situação ativa ou em afastamento dos senadores",}

        coef = nx.attribute_assortativity_coefficient(graph, atributo)
        matriz = nx.attribute_mixing_matrix(graph, atributo, mapping=mapeamento[atributo])

        children = []

        textual = [html.H4("Homofilia por Assortatividade"),
                   html.P("Análise de assortividade por {}".format(descricao[atributo])),
                   html.P("Coeficiente de Assortatividade: {:.4f}".format(coef)),]
        children += [html.Div(textual, style={'width': '40%', 'float': 'left', 'display': 'inline-block'})]

        labels = list(mapeamento[atributo].keys())
        fig = go.Figure(
            data=[go.Heatmap(z=matriz, x=labels, y=labels)],
            layout=go.Layout(
                title=go.layout.Title(text="Matriz de Mixagem"),
                height=350
            )
        )
        children += [html.Div([dcc.Graph(figure=fig)], style={'width': '49%', 'display': 'inline-block'})]

        return children


def converte_grafo_df(G, df_parlamentares):
    novas_colunas = ['pos_x', 'pos_y', 'adjacencias']
    df_arestas = pd.DataFrame(columns=['id_entrada', 'id_saida'])

    pos_x = []
    pos_y = []
    adjacencias = []
    id_entrada = []
    id_saida = []

    for edge in G.edges():
        id_entrada.append(edge[0])
        id_saida.append(edge[1])

    pos = nx.circular_layout(G)

    for node in G.nodes():
        x, y = pos[node]
        pos_x.append(x)
        pos_y.append(y)

    for node, adjacencies in enumerate(G.adjacency()):
        adjacencias.append(adjacencies[1])

    df_parlamentares['pos_x'] = pos_x
    df_parlamentares['pos_y'] = pos_y
    df_parlamentares['adjacencias'] = adjacencias
    df_arestas['id_entrada'] = id_entrada
    df_arestas['id_saida'] = id_saida

    return df_parlamentares, df_arestas

def cria_trace(df_parlamentares, df_arestas, cores, ano=None):
    edge_x = []
    edge_y = []
    for i in range(len(df_arestas)):
        x0 = df_parlamentares.iloc[df_arestas.iloc[i]['id_entrada']]['pos_x']
        y0 = df_parlamentares.iloc[df_arestas.iloc[i]['id_entrada']]['pos_y']
        x1 = df_parlamentares.iloc[df_arestas.iloc[i]['id_saida']]['pos_x']
        y1 = df_parlamentares.iloc[df_arestas.iloc[i]['id_saida']]['pos_y']
        inicio = [x0, y0]
        fim = [x1, y1]
        if inicio == fim:
            continue
        edge_x, edge_y = addEdge(inicio, fim, edge_x, edge_y, .8, 'end', .04, 30)
        '''edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)'''

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for i in range(len(df_parlamentares)):
        x = df_parlamentares.iloc[i]['pos_x']
        y = df_parlamentares.iloc[i]['pos_y']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text')
    
    if cores == 'conexoes':
        node_trace.marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Número de conexões',
                xanchor='left',
                titleside='right'
            ),
            line_width=2)

    cores_dos_nos = cores
    node_trace.marker=dict(
        reversescale=True,
        color=cores_dos_nos,
        size=10,
        line_width=2)

    node_text = []

    node_adjacencies = []

    for adjacencias in df_parlamentares['adjacencias'].tolist():
        if cores == 'conexoes':
            node_adjacencies.append(len(adjacencias))
    for node in range(len(df_parlamentares)):
        node_text.append(str(df_parlamentares.iloc[node].tratamento) + \
                         str(df_parlamentares.iloc[node].nome) )
        if not pd.isnull(df_parlamentares.iloc[node].email):
            node_text[-1] = node_text[-1] + \
                            '<br>e-mail: ' + str(df_parlamentares.iloc[node].email)# + \
                            #'<br>UF: ' + str(df_parlamentares.iloc[node].uf)
            if ano:
                partido = get_curr_partido(df_parlamentares.iloc[node].filiacoes, ano)
                node_text[-1] += '<br>Partido: ' + partido

    if cores == 'conexoes':
        node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    return edge_trace, node_trace

def cria_trace_bipartido(G, df_parlamentares, df_materias, pos, cores):
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

    node_x_senadores = []
    node_y_senadores = []
    node_x_materias = []
    node_y_materias = []
    for node in nx.get_node_attributes(G,'bipartite'):
        x, y = pos[node]
        if nx.get_node_attributes(G,'bipartite')[node] == 0:
            node_x_senadores.append(x)
            node_y_senadores.append(y)
        else:
            node_x_materias.append(x)
            node_y_materias.append(y)

    node_trace_senadores = go.Scatter(
        x=node_x_senadores, y=node_y_senadores,
        mode='markers',
        hoverinfo='text')

    if cores == 'conexoes':
        node_trace_senadores.marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Número de conexões',
                xanchor='left',
                titleside='right'
            ),
            line_width=2)
    elif cores == 'ativo':
        cores_dos_nos = colore_por_atributo(df_parlamentares, cores)
        node_trace_senadores.marker=dict(
            reversescale=True,
            color=cores_dos_nos,
            size=10,
            line_width=2)

    node_trace_materias = go.Scatter(
        x=node_x_materias, y=node_y_materias,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=5,
            line_width=2))

    node_text_senadores = []
    node_text_materias = []

    node_adjacencies = []

    for node, adjacencies in enumerate(G.adjacency()):
        if node in nx.get_node_attributes(G,'bipartite'):
            if nx.get_node_attributes(G,'bipartite')[node] == 0:
                if cores == 'afastado':
                    if df_parlamentares[df_parlamentares['cod'] == node].afastado.values[0] == 'Sim':
                        node_adjacencies.append(1)
                    else:
                        node_adjacencies.append(2)
                elif cores == 'conexoes':
                    node_adjacencies.append(len(adjacencies[1]))
                node_text_senadores.append(str(df_parlamentares[df_parlamentares['cod'] == node].tratamento.values[0]) + \
                                str(df_parlamentares[df_parlamentares['cod'] == node].nome.values[0]) )
                if pd.isnull(df_parlamentares[df_parlamentares['cod'] == node].email.values[0]) == False:
                    node_text_senadores[-1] = node_text_senadores[-1] + \
                                    '<br>e-mail: ' + str(df_parlamentares[df_parlamentares['cod'] == node].email.values[0])
            else:
                node_text_materias.append('Subtipo da matéria: ' + str(df_materias[df_materias['cod'] == node].nome_subtipo.values[0]))

    node_trace_senadores.marker.color = node_adjacencies
    node_trace_senadores.text = node_text_senadores
    node_trace_materias.text = node_text_materias

    return edge_trace, node_trace_senadores, node_trace_materias

def cria_pos_bipartido(G):
    u = []
    for key in nx.get_node_attributes(G,'bipartite'):
        if nx.get_node_attributes(G,'bipartite')[key] == 0:
            u.append(key)
    #u = [n for n in G.nodes if nx.get_node_attributes(G,'bipartite')[n] == 0]
    l, r = nx.bipartite.sets(G, top_nodes=u)
    pos = {}

    # Update position for node from each group
    # TODO: Melhorar posicionamento. Podemos espaçar e centralizar
    pos.update((node, (index, 2)) for index, node in enumerate(l))
    pos.update((node, (index, 1)) for index, node in enumerate(r))

    return pos

def converte_grafo_df_bipartido(G, df_parlamentares, df_materias, pos):
    novas_colunas = ['pos_x', 'pos_y', 'adjacencias']
    df_arestas = pd.DataFrame(columns=['id_entrada', 'id_saida'])

    pos_x_senador = []
    pos_y_senador = []
    adjacencias_senador = []

    pos_x_materia = []
    pos_y_materia = []
    adjacencias_materia = []

    id_entrada = []
    id_saida = []

    for edge in G.edges():
        id_entrada.append(edge[0])
        id_saida.append(edge[1])

    for node in G.nodes():
        x, y = pos[node]
        if nx.get_node_attributes(G,'bipartite')[node] == 0:
            pos_x_senador.append(x)
            pos_y_senador.append(y)
        else:
            pos_x_materia.append(x)
            pos_y_materia.append(y)

    for node, adjacencies in enumerate(G.adjacency()):
        if nx.get_node_attributes(G,'bipartite')[node] == 0:
            adjacencias_senador.append(adjacencies[1])
        else:
            adjacencias_materia.append(adjacencies[1])

    df_parlamentares['pos_x'] = pos_x_senador
    df_parlamentares['pos_y'] = pos_y_senador
    df_parlamentares['adjacencias'] = adjacencias_senador

    df_materias['pos_x'] = pos_x_materia
    df_materias['pos_y'] = pos_y_materia
    df_materias['adjacencias'] = adjacencias_materia

    df_arestas['id_entrada'] = id_entrada
    df_arestas['id_saida'] = id_saida

    return df_parlamentares, df_materias, df_arestas

def plotta_grafo(edge_trace, node_trace):
    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>Rede de Votações de Matérias de Parlamentares',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    #annotations=[ dict(
                    #    showarrow=False,
                    #    xref="paper", yref="paper",
                    #    x=0.005, y=-0.002 ) ],
                    height=750,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

def plotta_grafo_bipartido(edge_trace, node_trace_senadores, node_trace_materias):
    fig = go.Figure(data=[edge_trace, node_trace_senadores, node_trace_materias],
                layout=go.Layout(
                    title='<br>Rede de Votações de Matérias de Parlamentares',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    #annotations=[ dict(
                    #    showarrow=False,
                    #    xref="paper", yref="paper",
                    #    x=0.005, y=-0.002 ) ],
                    height=750,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

@app.callback(
    [Output('graph', 'figure'),
     Output('detalhes-analises', 'children')],
    [Input('botao-gerar-grafo', 'n_clicks')],
    [State('tipo-rede', 'value'),
     State('filtro-senadores', 'value'),
     State('coloracao-senadores', 'value'),
     State('filtrar-senadores', 'value'),
     State('senadores-checklist', 'value'),
     State('partidos-checklist', 'value'),
     State('alinhamentos-checklist', 'value'),
     State('materias-checklist', 'value'),
     State('filtrar-tempo', 'value'),
     State('selecao-ano', 'value'),
     State('selecao-mandato', 'value'),
     State('selecao-analise', 'value'),]
)
def gera_nova_rede(n_cliques, tipo_rede, filtro_senadores, coloracao_nos, filtrar_senadores, senadores_checklist,\
                   partidos_checklist, alinhamentos_checklist, materias_checklist, filtrar_tempo, filtro_ano, filtro_mandato,\
                   metrica_analise):
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

    df_materias_comum_filtro = global_store_materias_comum()
    df_materias_filtro = global_store_materias()

    if materias_checklist is not None:
        df_materias_comum_filtro = filtra_materias_subtipo(df_materias_comum_filtro, materias_checklist)
        df_materias_filtro = filtra_materias_subtipo(df_materias_filtro, materias_checklist)

    if tipo_rede == 'favor' or tipo_rede == 'contra':
        # Filtro temportal
        if filtrar_tempo == 'tempo-ano':
            df_materias_comum_filtro = filtrar_por_ano(df_materias_comum_filtro, filtro_ano)
        elif filtrar_tempo == 'tempo-mandato':
            df_materias_comum_filtro = filtrar_por_mandato(df_materias_comum_filtro, filtro_mandato)
        
        A = cria_mat_adj_votos(df_parlamentares_filtro, df_materias_comum_filtro, tipo_rede)
        
        if filtrar_senadores is not None and len(filtrar_senadores) > 0 and filtrar_senadores[0] == 'filtrar':
            A, df_parlamentares_filtro = filtra_mat_adj_votos(A, df_parlamentares_filtro)
        G = nx.from_numpy_matrix(A)

        df_parlamentares_filtro, df_arestas = converte_grafo_df(G, df_parlamentares_filtro)

        cores_dos_nos = None
        ano = None

        if coloracao_nos == 'ativo':
            cores_dos_nos, atr_dict = colore_por_afastamento(df_parlamentares_filtro)
            if metrica_analise == 'analise-assortatividade':
                nx.set_node_attributes(G, atr_dict, name="Ativo/Afastado")

        elif coloracao_nos == 'partido' and filtrar_tempo in ['tempo-ano', 'tempo-mandato']:
            ano = filtro_ano
            if filtrar_tempo == 'tempo-mandato':
                ano = filtro_mandato
            cores_dos_nos, atr_dict = colore_por_partido(df_parlamentares_filtro, ano)
            if metrica_analise == 'analise-assortatividade':
                nx.set_node_attributes(G, atr_dict, name="Partido")

        edge_trace, node_trace = cria_trace(df_parlamentares_filtro, df_arestas, cores_dos_nos, ano)
        analise = analisar_grafo(G, metrica_analise, atributo="Ativo/Afastado")
        fig = plotta_grafo(edge_trace, node_trace)
        
    elif tipo_rede == 'favor-bipartido' or tipo_rede == 'contra-bipartido':
        # Filtro temportal
        if filtrar_tempo == 'tempo-ano':
            df_materias_filtro = filtrar_por_ano(df_materias_filtro, filtro_ano)
        elif filtrar_tempo == 'tempo-mandato':
            df_materias_filtro = filtrar_por_mandato(df_materias_filtro, filtro_mandato)

        G = cria_grafo_bipartido(df_parlamentares_filtro, df_materias_filtro, tipo_rede)

        if filtrar_senadores is not None and len(filtrar_senadores) > 0 and filtrar_senadores[0] == 'filtrar':
            A = nx.adjacency_matrix(G)
            A = A.todense()
            A, df_parlamentares_filtro, df_materias_filtro = filtra_mat_adj_bipartido(A, df_parlamentares_filtro, df_materias_filtro)
            G = cria_grafo_bipartido(df_parlamentares_filtro, df_materias_filtro, tipo_rede)

        pos = cria_pos_bipartido(G)
        edge_trace, node_trace_senadores, node_trace_materias = cria_trace_bipartido(G, df_parlamentares_filtro, df_materias_filtro, pos, coloracao_nos)
        fig = plotta_grafo_bipartido(edge_trace, node_trace_senadores, node_trace_materias)
        analise = []
        
    fig.update_layout(transition_duration=500)
    print('update')
    return [fig, analise]

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