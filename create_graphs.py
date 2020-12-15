from collections import Counter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import pandas as pd
import math

df_partidos = pd.read_csv('./dataset/csvs/partidos.csv')
df_materias = pd.read_csv('./dataset/csvs/materias.csv')
df_materias_comum = pd.read_csv('./dataset/csvs/materias_comum.csv')
df_parlamentares = pd.read_csv('./dataset/csvs/parlamentares.csv')

def clean_alt_list(list_):
    list_ = list_.replace(', ', '","')
    list_ = list_.replace('[', '["')
    list_ = list_.replace(']', '"]')
    return list_

def limpa_colunas(df, coluna):
    df[coluna] = df[coluna].apply(clean_alt_list)
    df[coluna] = df[coluna].apply(eval)
    return df

df_parlamentares = limpa_colunas(df_parlamentares, 'telefones')
df_parlamentares = limpa_colunas(df_parlamentares, 'filiacoes')
df_parlamentares = limpa_colunas(df_parlamentares, 'votos_favor')
df_parlamentares = limpa_colunas(df_parlamentares, 'votos_contra')
df_materias_comum = limpa_colunas(df_materias_comum, 'outros_autores')

print(df_materias_comum.data_votacao)

#----------CREATE ADJASCENCY MATRIXES AND GRAPHS-----------

def extrai_listas(df_parlamentares, df_materias_comum):
    parlamentares = df_parlamentares['cod'].tolist()
    materias = df_materias_comum['cod'].tolist()
    return parlamentares, materias

def cria_mat_adj_votos(df_parlamentares, df_materias_comum, parlamentares, materias, tipo_voto='votos_favor'):
    n = len(df_parlamentares)
    A = np.zeros((n,n))
    if tipo_voto != 'votos_favor' and tipo_voto != 'votos_contra':
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
                index_parlamentar = parlamentares.index(autor_principal)
                A[i,index_parlamentar] += 1
            outros_autores = df_materias_comum.iloc[index_materia]['outros_autores']
            for autor in outros_autores:
                autor = autor.replace('\'', '')
                if autor == '':
                    continue
                autor = int(autor)
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

def make_label_dict(items, keys):
    label_dict = {}
    for i in range(len(keys)):
        label_dict[keys[i]] = items[i]
    return label_dict

def colore_por_afastamento(df_parlamentares, parlamentares):
    colors_node = []
    for i in range(len(parlamentares)):
        if (df_parlamentares[df_parlamentares['cod'] == parlamentares[i]].afastado == "Sim").all():
            colors_node.append('darksalmon')
        else:
            colors_node.append('darkblue')
    return colors_node

'''def plot_graph(G):
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
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>Network graph made with Python',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()'''

def cria_trace(G, df_parlamentares, parlamentares, cores):
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
        if df_parlamentares[df_parlamentares['cod'] == parlamentares[node]].afastado.item() == 'Sim':
            node_adjacencies.append(1)
        else:
            node_adjacencies.append(2)
        node_text.append(str(df_parlamentares[df_parlamentares['cod'] == parlamentares[node]].tratamento.item()) + \
                        str(df_parlamentares[df_parlamentares['cod'] == parlamentares[node]].nome.item()) )
        if pd.isnull(df_parlamentares[df_parlamentares['cod'] == parlamentares[node]].email.item()) == False:
            node_text[-1] = node_text[-1] + '<br>email: ' + str(df_parlamentares[df_parlamentares['cod'] == parlamentares[node]].email.item())

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
    fig.show()

parlamentares, materias = extrai_listas(df_parlamentares, df_materias_comum)

A_favor = cria_mat_adj_votos(df_parlamentares, df_materias_comum, parlamentares, materias, 'votos_favor')
A_contra = cria_mat_adj_votos(df_parlamentares, df_materias_comum, parlamentares, materias, 'votos_contra')

A_favor_filtro, parlamentares_favor = filtra_mat_adj_votos(A_favor, parlamentares)
A_contra_filtro, parlamentares_contra = filtra_mat_adj_votos(A_contra, parlamentares)

df_parlamentares_favor = filtra_df_parlamentares(df_parlamentares, parlamentares_favor)
df_parlamentares_contra = filtra_df_parlamentares(df_parlamentares, parlamentares_contra)

label_dict_favor = make_label_dict(parlamentares_favor, range(len(parlamentares_favor)))
label_dict_contra = make_label_dict(parlamentares_contra, range(len(parlamentares_contra)))

cores_nos_favor = colore_por_afastamento(df_parlamentares, parlamentares_favor)
cores_nos_contra = colore_por_afastamento(df_parlamentares, parlamentares_contra)

#colors_edge_1 = ['b']*n
#colors_edge_2 = ['r']*n

G_favor_filtro = nx.from_numpy_matrix(A_favor_filtro)
G_contra_filtro = nx.from_numpy_matrix(A_contra_filtro)

for node, adjacencies in enumerate(G_favor_filtro.adjacency()):
        print(node)
        print(adjacencies)

edge_trace_favor, node_trace_favor = cria_trace(G_favor_filtro, df_parlamentares, parlamentares_favor, cores_nos_favor)
edge_trace_contra, node_trace_contra = cria_trace(G_contra_filtro, df_parlamentares, parlamentares_contra, cores_nos_contra)

file1 = open('edge_trace_favor', 'wb')
file2 = open('node_trace_favor', 'wb')

pickle.dump(edge_trace_favor, file1)
pickle.dump(node_trace_favor, file2)

file1.close()
file2.close()

file1 = open('edge_trace_contra', 'wb')
file2 = open('node_trace_contra', 'wb')

pickle.dump(edge_trace_contra, file1)
pickle.dump(node_trace_contra, file2)

file1.close()
file2.close()

plotta_grafo(edge_trace_favor, node_trace_favor)
plotta_grafo(edge_trace_contra, node_trace_contra)

'''nx.draw(G_favor_filtro, node_size=250, labels=label_dict_favor, node_color=cores_nos_favor, with_labels=True)
plt.show()
nx.draw(G_contra_filtro, node_size=250, labels=label_dict_contra, node_color=cores_nos_contra, with_labels=True)
plt.show()

file1 = open('grafo_favor_filtro', 'wb')
file2 = open('grafo_contra_filtro', 'wb')

pickle.dump(G_favor_filtro, file1)
pickle.dump(G_contra_filtro, file2)

file1.close()
file2.close()'''





'''nx.draw(G1, node_size=250, labels=label_dict, edge_color=colors_edge_1, node_color=colors_node, with_labels=True)
plt.show()

nx.draw(G2, node_size=250, labels=label_dict, edge_color=colors_edge_2, node_color=colors_node, with_labels=True)
plt.show()

nx.draw(G3, node_size=250, labels=label_dict, edge_color=colors_edge_1, with_labels=True)
plt.show()'''