import xml.etree.ElementTree as et
from collections import Counter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from analises import *

lista_atuais=["4981", "5982", "945", "5967", "5529", "6295", "5936", "5990", "5973", "739", "475", "3830", "5996", "5537", "4994", "5976", "3777", "5718", "5531", "22",\
        "5953", "5540", "4525", "345", "5894", "5008", "5385", "4770", "35", "581", "4545", "4531", "5627", "5895", "5350", "3361", "90", "1249", "5533", "5979", "5926", "1186", "5585", \
        "5557", "5666", "5376", "742", "285", "5422", "5942", "1023", "6027", "5985", "63", "5525", "5924", "5523", "825", "374", "5502", "5012", "5236", "70", "677", "5905", "5732", \
        "5352", "5322", "2331", "5527", "5988", "5959", "4560", "3396", "5535", "5899", "1173", "5411", "5783", "3806"]

lista_afastados=["751", "470", "5998", "5637", "3713", "5615", "4558", "5639", "5849", "878", "5929", "151", "5617",\
        "5927", "4606", "5971", "5561", "5635", "456", "5621", "6005", "5619", "5748", "4786"]

attributes_parlamentares = ['IdentificacaoParlamentar', 'MandatoAtual',\
    'FiliacaoAtual', 'MembroAtualComissoes', 'CargosAtuais', 'LiderancasAtuais',\
    'MateriasDeAutoriaTramitando', 'RelatoriasAtuais', 'OutrasInformacoes']

IdentificacaoParlamentar = ['CodigoParlamentar', 'CodigoPublicoNaLegAtual', 'NomeParlamentar', 'NomeCompletoParlamentar',\
    'SexoParlamentar', 'FormaTratamento', 'UrlFotoParlamentar', 'UrlPaginaParlamentar',\
    'UrlPaginaParticular', 'EmailParlamentar', 'SiglaPartidoParlamentar', 'UfParlamentar']

MandatoAtual = ['CodigoMandato', 'UfParlamentar', 'PrimeiraLegislaturaDoMandato',\
        'SegundaLegislaturaDoMandato', 'DescricaoParticipacao', 'Suplentes', 'Exercicios']

FiliacaoAtual = ['Partido', 'DataFiliacao']

colunas_parlamentares = ['CodigoParlamentar', 'CodigoPublicoNaLegAtual', 'NomeCompletoParlamentar',\
    'SexoParlamentar', 'FormaTratamento', 'UrlFotoParlamentar', 'UrlPaginaParlamentar',\
    'UrlPaginaParticular', 'EmailParlamentar', 'SiglaPartidoParlamentar', 'UfParlamentar']

attributes_autorias = ['IdentificacaoMateria', 'EmentaMateria', 'SituacaoAtual']

IdentificacaoMateria = ['CodigoMateria', 'SiglaSubtipoMateria', 'DescricaoSubtipoMateria', 'AnoMateria', 'IndicadorTramitando']

attributes_votacoes = ['IdentificacaoMateria', 'IndicadorVotacaoSecreta', 'DescricaoVotacao', 'DescricaoResultado', 'SiglaDescricaoVoto']

#só vamos pegar as matérias que não estão mais tramitando e não são por voto secreto
colunas_materias = ['CodigoMateria', 'SiglaSubtipoMateria', 'DescricaoSubtipoMateria', 'AnoMateria', 'EmentaMateria', 'CodigoAutorPrincipal', 'DescricaoVotacao', 'DescricaoResultado', 'SiglaDescricaoVoto']

partidos_atuais = []

for codigo in lista_atuais:
    print(codigo)
    xtree = et.parse("./dataset/atuais/dados_" + codigo + ".xml")
    xroot = xtree.getroot()
    filiacao = xroot[1].find('FiliacaoAtual')
    sigla = xroot[1].find('FiliacaoAtual').find('Partido').find('SiglaPartido').text
    partidos_atuais.append(sigla)

print(partidos_atuais)

partidos_unicos = list(Counter(partidos_atuais).keys())

print(partidos_unicos)
print("{} partidos".format(len(partidos_unicos)))

dic_partidos = {
  'PDT' : 'oposicao',
  'CIDADANIA' : 'medio',
  'PODEMOS' : 'medio',
  'PSD' : 'apoio',
  'PP' : 'apoio',
  'MDB' : 'apoio',
  'DEM' : 'apoio',
  'REDE' : 'oposicao',
  'PROS' : 'medio',
  'REPUBLICANOS' : 'apoio',
  'PT' : 'oposicao',
  'PSDB' : 'apoio',
  'PL' : 'apoio',
  'PSB' : 'oposicao',
  'PSL' : 'apoio',
  'PSC' : 'apoio'
}

dic_cores = {
    'oposicao' : 'r',
    'medio' : 'y',
    'apoio' : 'g'
}

materias_votacoes = []

for codigo in lista_atuais:
    print(codigo)
    xtree = et.parse("./dataset/votacoes/dados_" + codigo + "_votacoes.xml")
    xroot = xtree.getroot()
    for node in xroot[1][1]:
        voto_secreto = node.find('IndicadorVotacaoSecreta').text
        if voto_secreto == "Sim":
            continue
        materia = node[1]
        cd = materia.find('CodigoMateria')
        if cd is None:
            continue
        cd_materia = cd.text
        if cd_materia not in materias_votacoes:
            materias_votacoes.append(cd_materia)

materias_comum = []
parlamentares_comum = []

for codigo in lista_atuais:
    print(codigo)
    xtree = et.parse("./dataset/autorias/dados_" + codigo + "_autorias.xml")
    xroot = xtree.getroot()
    for node in xroot[1][1]:
        materia = node[0][0]
        cd = materia.find('CodigoMateria')
        if cd is None:
            continue
        cd_materia = cd.text
        if cd_materia not in materias_votacoes:
            continue
        autor_principal = node.find('IndicadorAutorPrincipal').text
        if autor_principal == "Sim":
            materias_comum.append(cd_materia)
            parlamentares_comum.append(codigo)

print(materias_comum)
print("qtd materias: ", len(materias_comum))

print(parlamentares_comum)
parlamentares_unicos = list(Counter(parlamentares_comum).keys())
print("qtd parlamentares: ", len(parlamentares_unicos))

n = len(lista_atuais)
A1 = np.zeros((n,n))
A2 = np.zeros((n,n))

for i in range(n):
    xtree = et.parse("./dataset/votacoes/dados_" + lista_atuais[i] + "_votacoes.xml")
    xroot = xtree.getroot()
    for node in xroot[1][1]:
        materia = node[1]
        cd = materia.find('CodigoMateria')
        if cd is None:
            continue
        cd_materia = cd.text
        if cd_materia not in materias_comum:
            continue
        index_materia = materias_comum.index(cd_materia)
        cd_autor = parlamentares_comum[index_materia]
        j = lista_atuais.index(cd_autor)
        voto = node.find('SiglaDescricaoVoto').text
        if voto == "Sim":
            A1[i,j] += 1
        elif voto == "Não":
            A2[i,j] += 1

print(A1)
print(A2)

def make_label_dict(items, keys):
    label_dict = {}
    for i in range(len(keys)):
        label_dict[keys[i]] = items[i]
    return label_dict

label_dict = make_label_dict(lista_atuais, range(n))

colors_edge_1 = 'b'
colors_edge_2 = 'r'

colors_node = []

for i in range(len(lista_atuais)):
    colors_node.append(dic_cores[dic_partidos[partidos_atuais[i]]])

G1 = nx.from_numpy_matrix(A1)
G2 = nx.from_numpy_matrix(A2)

nx.draw(G1, node_size=250, labels=label_dict, edge_color=colors_edge_1, node_color=colors_node, with_labels=True)
plt.show()

f = open("dataInfo/G1_info.txt", "w")
deg = degree_analysis(G1)
f.write(f"Grau: "
        f"\tMínimo: {deg[0]}\n"
        f"\tMáximo: {deg[1]}\n"
        f"\tMédia: {deg[2]} +- {deg[3]}\n"
        f"\tMediana: {deg[4]}\n"
        f"{deg[5]} (PDF)\n\n")

dist = distance_analysis(G1)
f.write(f"Distância: ({dist[6]} pares alcançáveis)\n"
        f"\tMínima: {dist[0]}\n"
        f"\tMáxima: {dist[1]}\n"
        f"\tMédia: {dist[2]} +- {dist[3]}\n"
        f"\tMediana: {dist[4]}\n"
        f"{dist[5]} (PDF)\n\n")

con = connexity_analysis(G1)
f.write(f"Componentes conexas: {con[0]} componentes independentes \n"
        f"\tMínima: {con[1]}\n"
        f"\tMáxima: {con[2]}\n"
        f"\tMédia: {con[3]} +- {con[4]}\n"
        f"\tMediana: {con[5]}\n"
        f"{con[6]} (PDF)\n\n")

bet = betweenness_analysis(G1)
f.write(f"Betweenness: \t(Contralidade 1)\n"
        f"\tMínimo: {bet[0]}\n"
        f"\tMáximo: {bet[1]}\n"
        f"\tMédia: {bet[2]} +- {bet[3]}\n"
        f"\tMediana: {bet[4]}\n"
        f"{bet[5]} (CCDF)\n\n")

close = closeness_analysis(G1)
f.write(f"Closeness: \t(Contralidade 2)\n"
        f"\tMínimo: {close[0]}\n"
        f"\tMáximo: {close[1]}\n"
        f"\tMédia: {close[2]} +- {close[3]}\n"
        f"\tMediana: {close[4]}\n"
        f"{close[5]} (CCDF)\n\n")

clust = clustering_analysis(G1)
f.write(f"Clustering: \n"
        f"\tMínimo: {clust[0]}\n"
        f"\tMáximo: {clust[1]}\n"
        f"\tMédia: {clust[2]} +- {clust[3]}\n"
        f"\tMediana: {clust[4]}\n"
        f"{clust[5]} (CCDF)\n")

nx.draw(G2, node_size=250, labels=label_dict, edge_color=colors_edge_2, node_color=colors_node, with_labels=True)
plt.show()

f = open("dataInfo/G2_info.txt", "w")
deg = degree_analysis(G2)
f.write(f"Grau: "
        f"\tMínimo: {deg[0]}\n"
        f"\tMáximo: {deg[1]}\n"
        f"\tMédia: {deg[2]} +- {deg[3]}\n"
        f"\tMediana: {deg[4]}\n"
        f"{deg[5]} (PDF)\n\n")

dist = distance_analysis(G2)
f.write(f"Distância: ({dist[6]} pares alcançáveis)\n"
        f"\tMínima: {dist[0]}\n"
        f"\tMáxima: {dist[1]}\n"
        f"\tMédia: {dist[2]} +- {dist[3]}\n"
        f"\tMediana: {dist[4]}\n"
        f"{dist[5]} (PDF)\n\n")

con = connexity_analysis(G2)
f.write(f"Componentes conexas: {con[0]} componentes independentes \n"
        f"\tMínima: {con[1]}\n"
        f"\tMáxima: {con[2]}\n"
        f"\tMédia: {con[3]} +- {con[4]}\n"
        f"\tMediana: {con[5]}\n"
        f"{con[6]} (PDF)\n\n")

bet = betweenness_analysis(G2)
f.write(f"Betweenness: \t(Contralidade 1)\n"
        f"\tMínimo: {bet[0]}\n"
        f"\tMáximo: {bet[1]}\n"
        f"\tMédia: {bet[2]} +- {bet[3]}\n"
        f"\tMediana: {bet[4]}\n"
        f"{bet[5]} (CCDF)\n\n")

close = closeness_analysis(G2)
f.write(f"Closeness: \t(Contralidade 2)\n"
        f"\tMínimo: {close[0]}\n"
        f"\tMáximo: {close[1]}\n"
        f"\tMédia: {close[2]} +- {close[3]}\n"
        f"\tMediana: {close[4]}\n"
        f"{close[5]} (CCDF)\n\n")

clust = clustering_analysis(G2)
f.write(f"Clustering: \n"
        f"\tMínimo: {clust[0]}\n"
        f"\tMáximo: {clust[1]}\n"
        f"\tMédia: {clust[2]} +- {clust[3]}\n"
        f"\tMediana: {clust[4]}\n"
        f"{clust[5]} (CCDF)\n")

materias_unicas = []
parlamentares_materias = []

for codigo in lista_atuais:
    print(codigo)
    xtree = et.parse("./dataset/autorias/dados_" + codigo + "_autorias.xml")
    xroot = xtree.getroot()
    for node in xroot[1][1]:
        materia = node[0][0]
        cd = materia.find('CodigoMateria')
        if cd is None:
            continue
        outros_autores = node.find('IndicadorOutrosAutores')
        if outros_autores != None:
            text = outros_autores.text
            if text == 'Não':
                continue
        cd_materia = cd.text
        if cd_materia not in materias_unicas:
            materias_unicas.append(cd_materia)
            parlamentares_materias.append([codigo])
        else:
            index = materias_unicas.index(cd_materia)
            parlamentares_materias[index].append(codigo)

print("materias unicas: ", materias_unicas)
print("parlamentares materias: ", parlamentares_materias)

A3 = np.zeros((n,n))

for i in range(len(materias_unicas)):
    for j in range(len(parlamentares_materias[i])):
        for k in range(j+1, len(parlamentares_materias[i])):
            origin_index = lista_atuais.index(parlamentares_materias[i][j])
            dest_index = lista_atuais.index(parlamentares_materias[i][k])
            A3[origin_index,dest_index] += 1
            A3[dest_index,origin_index] += 1

print(A3)

G3 = nx.from_numpy_matrix(A3)

nx.draw(G3, node_size=250, labels=label_dict, edge_color=colors_edge_1, with_labels=True)
plt.show()

f = open("./dataInfo/G3_info.txt", "w")
deg = degree_analysis(G3)
f.write(f"Grau: "
        f"\tMínimo: {deg[0]}\n"
        f"\tMáximo: {deg[1]}\n"
        f"\tMédia: {deg[2]} +- {deg[3]}\n"
        f"\tMediana: {deg[4]}\n"
        f"{deg[5]} (PDF)\n\n")

dist = distance_analysis(G3)
f.write(f"Distância: ({dist[6]} pares alcançáveis)\n"
        f"\tMínima: {dist[0]}\n"
        f"\tMáxima: {dist[1]}\n"
        f"\tMédia: {dist[2]} +- {dist[3]}\n"
        f"\tMediana: {dist[4]}\n"
        f"{dist[5]} (PDF)\n\n")

con = connexity_analysis(G3)
f.write(f"Componentes conexas: {con[0]} componentes independentes \n"
        f"\tMínima: {con[1]}\n"
        f"\tMáxima: {con[2]}\n"
        f"\tMédia: {con[3]} +- {con[4]}\n"
        f"\tMediana: {con[5]}\n"
        f"{con[6]} (PDF)\n\n")

bet = betweenness_analysis(G3)
f.write(f"Betweenness: \t(Contralidade 1)\n"
        f"\tMínimo: {bet[0]}\n"
        f"\tMáximo: {bet[1]}\n"
        f"\tMédia: {bet[2]} +- {bet[3]}\n"
        f"\tMediana: {bet[4]}\n"
        f"{bet[5]} (CCDF)\n\n")

close = closeness_analysis(G3)
f.write(f"Closeness: \t(Contralidade 2)\n"
        f"\tMínimo: {close[0]}\n"
        f"\tMáximo: {close[1]}\n"
        f"\tMédia: {close[2]} +- {close[3]}\n"
        f"\tMediana: {close[4]}\n"
        f"{close[5]} (CCDF)\n\n")

clust = clustering_analysis(G3)
f.write(f"Clustering: \n"
        f"\tMínimo: {clust[0]}\n"
        f"\tMáximo: {clust[1]}\n"
        f"\tMédia: {clust[2]} +- {clust[3]}\n"
        f"\tMediana: {clust[4]}\n"
        f"{clust[5]} (CCDF)\n")