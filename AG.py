# Leitura dos dados de entrada: Não precisamos alterar essa parte, pois  já tem uma função 
# ler_arquivo que faz a leitura dos dados de entrada.

# Armazenamento dos dados de entrada: Também não precisamos fazer alterações aqui, 
# pois já tem a classe Dados para armazenar os dados do problema.

# Estrutura para representação (armazenamento) de uma solução: No algoritmo genético, a solução será uma lista de números inteiros representando as colunas selecionadas. 
# Portanto, não precisamos de uma estrutura específica para armazenar a solução além da lista de índices das colunas selecionadas.

# Estrutura para armazenamento da população de soluções: Precisaremos de uma lista de soluções para representar a população.
# Cada solução será uma lista de índices das colunas selecionadas.

# O Algoritmo Genético:
    # Função de Seleção das soluções para cruzamento: vai ser o método da roleta viciada, onde a probabilidade de seleção de cada solução é proporcional à sua aptidão.
    # Função para cruzamento das soluções: Vamos implementar o crossover de um ponto, onde um ponto de corte é escolhido aleatoriamente e as partes das soluções pais são trocadas.
    # Função de mutação: Implementaremos a mutação por inversão, onde escolhemos dois pontos de corte e invertemos a ordem dos genes entre esses pontos.
    # Procedimento de busca local: Como você já temos uma função de melhoramento, vamos usá-la como procedimento de busca local após a etapa de mutação.
    # Definir e implementar o critério de parada do algoritmo genético: Vamos utilizar um número máximo de gerações como critério de parada.


from copy import deepcopy
import math
import random
from math import inf
import sys
from timeit import default_timer as timer

# Classe que representa uma coluna de dados contendo (numero da coluna, custo, linhas que a coluna cobre)
class Coluna:
    def __init__(self, indice: int, custo: float, linhascobertas: set[int]):
        self.indice = indice
        self.custo = custo
        self.linhascobertas = linhascobertas

# Classe que representa os dados do problema (numero de linhas, numero de colunas, colunas)
class Dados:
    def __init__(self, nlinhas: int, ncolunas: int, colunas: list[Coluna]):
        self.nlinhas = nlinhas
        self.ncolunas = ncolunas
        self.colunas = colunas

# Funções de custo gulosas para o algoritmo construtivo guloso do Set Covering Problem
funcoes_de_custo = [
    lambda cj, kj: cj,
    lambda cj, kj: cj / kj,
    lambda cj, kj: cj / math.log2(kj) if math.log2(kj) != 0 else inf,
    lambda cj, kj: cj / (kj * math.log2(kj)) if (kj * math.log2(kj)) != 0 else inf,
    lambda cj, kj: cj / (kj * math.log(kj)) if kj * math.log(kj) != 0 else inf,
    lambda cj, kj: cj / (kj * kj),
    lambda cj, kj: cj ** (1 / 2) / (kj * kj),
]

# Função que lê o arquivo de entrada e retorna os dados do problema
def ler_arquivo(arq: str) -> Dados:
    with open(arq, "r") as f:
        linhas = f.readlines()

    nmr_linhas = int(linhas[0].split()[1])
    nmr_colunas = int(linhas[1].split()[1])

    dados = []
    for linha in linhas[3:]:
        elementos = linha.split()
        indice = int(elementos[0])
        custo = float(elementos[1])
        linhas_cobertas = set([int(x) for x in elementos[2:]])
        dado = Coluna(indice, custo, linhas_cobertas)
        dados.append(dado)

    return Dados(nmr_linhas, nmr_colunas, dados)

# Função que retorna uma função de custo aleatória para o algoritmo construtivo guloso
def funcao_aleatoria(custo: float, kj: int) -> float:
    return random.choice(funcoes_de_custo)(custo, kj)

# Função que remove colunas redundantes da solução
def remove_colunas_redundantes(S, dados):
    T = S.copy()
    wi = [
        sum(1 for j in S if i in dados.colunas[j].linhascobertas)
        for i in range(1, dados.nlinhas + 1)
    ]
    while T:
        j = random.choice(list(T))
        T.remove(j)
        Bj = dados.colunas[j].linhascobertas
        if all(wi[i - 1] >= 2 for i in Bj):
            S.remove(j)
            for i in Bj:
                wi[i - 1] -= 1
    return S

# Função que implementa o algoritmo construtivo guloso para o Set Covering Problem
def construtivo(dados):
    solucao = set()
    R = set(range(1, dados.nlinhas + 1))

    Pj = [set() for _ in range(dados.ncolunas)]
    for j, coluna in enumerate(dados.colunas):
        Pj[j] = set(coluna.linhascobertas)

    while R != set():
        num_linhas_cobertas_por_coluna = [len(R.intersection(pj)) for pj in Pj]
        J = min(
            range(dados.ncolunas),
            key=lambda j: funcao_aleatoria(
                dados.colunas[j].custo, num_linhas_cobertas_por_coluna[j]
            )
            if num_linhas_cobertas_por_coluna[j] > 0
            else float("inf"),
        )

        R = R.difference(Pj[J])
        solucao.add(J)

    solucao = sorted(solucao, key=lambda j: dados.colunas[j].custo, reverse=True)

    for i in solucao:
        if set.union(*[Pj[j] for j in solucao if j != i]) == set(range(1, dados.nlinhas + 1)):
            solucao.remove(i)

    solucao = remove_colunas_redundantes(solucao, dados)

    assert valid_solution(solucao, dados)

    custo = sum([dados.colunas[j].custo for j in solucao])

    return solucao, custo

# Função que verifica se a solução encontrada é válida
def valid_solution(solucao, dados):
    rows = [0] * dados.nlinhas
    for c in solucao:
        for r in dados.colunas[c].linhascobertas:
            rows[r - 1] = 1
    return sum(rows) == dados.nlinhas

# Algoritmo genético combinado com busca local
def algoritmo_genetico_com_busca_local(dados, tamanho_populacao, num_geracoes, probabilidade_mutacao):

    start = timer()

    # Inicialização da população
    populacao = [construtivo(dados)[0] for _ in range(tamanho_populacao)]
    melhor_solucao = None
    melhor_custo = float('inf')

    for geracao in range(num_geracoes):
        nova_populacao = []

        # Seleção dos pais para o cruzamento
        pais_selecionados = selecao(populacao, dados)
        
        # Cruzamento
        for i in range(0, len(pais_selecionados), 2):
            pai1 = pais_selecionados[i]
            pai2 = pais_selecionados[i+1]
            filho1, filho2 = crossover_um_ponto(pai1, pai2)

            # Mutação
            filho1 = mutacao(filho1, probabilidade_mutacao)
            filho2 = mutacao(filho2, probabilidade_mutacao)

            nova_populacao.extend([filho1, filho2])

        # Busca local
        for i in range(len(nova_populacao)):
            nova_populacao[i] = busca_local(nova_populacao[i], dados)

        # Atualização da população
        populacao = nova_populacao

        # Atualização da melhor solução encontrada até o momento
        for solucao in populacao:
            custo = sum([dados.colunas[j].custo for j in solucao])
            if custo < melhor_custo:
                melhor_solucao = solucao
                melhor_custo = custo

    end = timer()

    tempoDeExecucao = end - start

    return melhor_solucao, melhor_custo, tempoDeExecucao

# Função de seleção dos pais para cruzamento (roleta viciada)
def selecao(populacao, dados):
    custos = [sum([dados.colunas[j].custo for j in solucao]) for solucao in populacao]
    soma_custos = sum(custos)
    probabilidades = [custo / soma_custos for custo in custos]
    pais_selecionados = random.choices(populacao, weights=probabilidades, k=len(populacao))
    return pais_selecionados

# Operador de crossover de um ponto
def crossover_um_ponto(pai1, pai2):
    ponto_corte = random.randint(1, min(len(pai1), len(pai2)) - 1)
    filho1 = pai1[:ponto_corte] + pai2[ponto_corte:]
    filho2 = pai2[:ponto_corte] + pai1[ponto_corte:]
    return filho1, filho2

# Operador de mutação por inversão
def mutacao(solucao, probabilidade_mutacao):
    if random.random() < probabilidade_mutacao:
        ponto1, ponto2 = random.sample(range(len(solucao)), 2)
        ponto1, ponto2 = min(ponto1, ponto2), max(ponto1, ponto2)
        solucao[ponto1:ponto2+1] = reversed(solucao[ponto1:ponto2+1])
    return solucao

def melhoramento(solucao, dados):
    d = 0
    D = math.ceil(random.uniform(0.05, 0.7) * len(solucao[0]))
    E = math.ceil(random.uniform(1.1, 2) * max(dados.colunas[j].custo for j in solucao[0]))

    # Quantidade de colunas que cobrem cada linha
    wi = [0] * dados.nlinhas
    for j in solucao[0]:
        for i in dados.colunas[j].linhascobertas:
            wi[i - 1] += 1

    # Conjunto de colunas que não estão na solução
    colunas_fora_da_solucao = set(range(dados.ncolunas)).difference(solucao[0])

    # Remove colunas da solução até que o número de colunas seja igual a D
    while d != D:
        k = random.choice(solucao[0])
        solucao[0].remove(k)
        colunas_fora_da_solucao.add(k)

        for i in dados.colunas[k].linhascobertas:
            wi[i - 1] -= 1
        d += 1

    U = set()
    for i in range(1, dados.nlinhas + 1):
        if wi[i - 1] == 0:
            U.add(i)

    # Adiciona colunas à solução até que todas as linhas sejam cobertas
    while U:

        # Lista de colunas que não estão na solução e que possuem custo menor ou igual a E
        Re = list(j for j in colunas_fora_da_solucao if dados.colunas[j].custo <= E)

        alpha_j = []
        for coluna in Re:
            # calcular quantas linhas não cobertas Re cobre
            vj = 0
            for linha in dados.colunas[coluna].linhascobertas:
                if linha in U:
                    vj += 1
            alpha_j.append(vj)

        beta_j = [
            (dados.colunas[j].custo / alpha_j[i]) if alpha_j[i] != 0 else inf
            for i, j in enumerate(Re)
        ]
        bmin = min(beta_j)
        K = set(j for j, beta_j in zip(Re, beta_j) if beta_j == bmin)

        j = random.choice(list(K))
        colunas_fora_da_solucao.remove(j)
        solucao[0].append(j)

        for i in dados.colunas[j].linhascobertas:
            wi[i - 1] += 1

        U = set()
        for i in range(1, dados.nlinhas + 1):
            if wi[i - 1] == 0:
                U.add(i)

    for k in reversed(solucao[0]):
        if all(wi[i - 1] > 1 for i in dados.colunas[k].linhascobertas):
            solucao[0].remove(k)
            colunas_fora_da_solucao.add(k)
            for i in dados.colunas[k].linhascobertas:
                wi[i - 1] -= 1

    solucao = list(solucao)
    solucao[1] = sum([dados.colunas[j].custo for j in solucao[0]])
    solucao = tuple(solucao)

    # Garante que a solução é válida
    assert valid_solution(solucao[0], dados)

    return solucao

# Função que implementa a busca local
def busca_local(solucao, dados):
    solucao = list(solucao)  # Convertendo para lista para facilitar a manipulação
    solucao, custo = melhoramento((solucao, sum([dados.colunas[j].custo for j in solucao])), dados)
    return solucao

def main():
    nome_do_arquivo = sys.argv[1]
    tamanho_populacao = int(sys.argv[2])
    num_geracoes = int(sys.argv[3])
    probabilidade_mutacao = float(sys.argv[4])

    dados = ler_arquivo(nome_do_arquivo)
    melhor_solucao, melhor_custo, tempo_execucao = algoritmo_genetico_com_busca_local(dados, tamanho_populacao, num_geracoes, probabilidade_mutacao)

    # Corrigindo a solução para somar 1 apenas aos valores
    melhor_solucao_corrigida = [coluna + 1 for coluna in melhor_solucao]

    print(f"Melhor solução encontrada: {melhor_solucao_corrigida}")
    print(f"Custo da melhor solução: {melhor_custo}")
    print(f"Tempo de execução: {tempo_execucao} segundos")

if __name__ == "__main__":
    main()
