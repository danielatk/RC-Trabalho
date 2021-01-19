# Trabalho Final de Redes Complexas

Aplicativo de visualização das redes de votação do Senado Federal. 
A página pode ser utilizada fazendo download deste repositório e 
rodando o arquivo ``app.py`` no terminal.

Atualmente há dois tipos de redes implementadas:

    1. Redes de senadores, com arestas direcionadas apontando do 
    senador votante para o senador autor primário de uma matéria
    2. Redes bipartidas de senadores e matérias, com as arestas 
    juntando o senador votante e a matéria
    
A página é capaz de plotar as redes de senadores, coloridos por 
seu caráter de atividade ou afastamento do posto. Diferentes 
redes podem ser geradas filtrando os nós e arestas por tempo, 
senadores, tipo de matéria, entre outros. Além disso, ela é 
capaz de realizar análises como as características da 
distribuição de grau e centralidade por PageRank. 

Um vídeo ilustrativo pode ser visto no arquivo 
[funcionamento_site](funcionamento_site.mov).