# Trabalho sobre o problema de cobertura de conjuntos (Set Covering Problem)
# Guilherme Frare Clemente - RA: 124349

# Modelagem e Otimização de Algoritmos

from copy import deepcopy
import math, random
from math import inf
import sys, time

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


# Função que lê o arquivo de entrada e retorna os dados do problema,
# Ou seja, retorna a classe dados, contendo o numero de linhas, o numero de colunas
# e as colunas(numero da coluna, custo, linhas cobertas)
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
        linhas_cobertas = [int(x) for x in elementos[2:]]
        dado = Coluna(indice, custo, linhas_cobertas)
        dados.append(dado)

    return Dados(nmr_linhas, nmr_colunas, dados)


# Função que retorna uma função de custo aleatória para o algoritmo construtivo guloso
def funcao_aleatoria(custo: float, kj: int) -> float:
    return random.choice(funcoes_de_custo)(custo, kj)


# Função que remove colunas redundantes da solução,
# Pois na maioria dos casos, a solução gulosa retorna colunas redundantes, ou seja,
# colunas que não são necessárias para cobrir todas as linhas (já possui outra coluna que cobre a mesma linha)
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
# Recebe como parâmetro os dados do problema e retorna a solução viável e o custo(da coluna) da solução
# A solução é uma lista de indices das colunas que foram escolhidas
# O custo é a soma dos custos das colunas escolhidas
# A solução é válida se todas as linhas forem cobertas
def construtivo(dados):
    solucao = set()
    R = set(range(1, dados.nlinhas + 1))

    Pj = [set() for _ in range(dados.ncolunas)]
    for j, coluna in enumerate(dados.colunas):
        Pj[j] = set(coluna.linhascobertas)

    # Faz a interseção de todas as linhas cobertas por cada coluna
    # E seleciona o menor custo entre as colunas que cobrem a linha
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

    # Ordena as colunas da solução por custo decrescente
    solucao = sorted(solucao, key=lambda j: dados.colunas[j].custo, reverse=True)

    for i in solucao:
        if set.union(*[Pj[j] for j in solucao if j != i]) == set(
            range(1, dados.nlinhas + 1)):
            solucao.remove(i)

    # Remove colunas redundantes da solução para melhorar o custo
    solucao = remove_colunas_redundantes(solucao, dados)

    # Garante que a solução é válida
    assert valid_solution(solucao, dados)

    # Calcula o custo da solução
    custo = sum([dados.colunas[j].custo for j in solucao])

    return solucao, custo


# Função que implementa outro algoritmo construtivo guloso para o Set Covering Problem
# Recebe como parâmetro os dados do problema e retorna a solução viável e o custo(da coluna) da solução
# A solução é uma lista de indices das colunas que foram escolhidas
# O custo é a soma dos custos das colunas escolhidas
# O algoritmo, em cada iteração, escolhe a coluna que cobre a maior quantidade de linhas ainda não cobertas
def construtivo2(dados):
    solucao = list()
    linhas_nao_cobertas = set(range(1, dados.nlinhas + 1))

    while linhas_nao_cobertas:
        # Calcula a quantidade de linhas não cobertas por cada coluna
        contagem_linhas_nao_cobertas = [
            sum(1 for i in linhas_nao_cobertas if i in dados.colunas[j].linhascobertas)
            for j in range(dados.ncolunas)
        ]

        # Escolhe a coluna que cobre o maior número de linhas não cobertas
        melhor_coluna = contagem_linhas_nao_cobertas.index(
            max(contagem_linhas_nao_cobertas)
        )

        # Adiciona a coluna escolhida à solução
        solucao.append(melhor_coluna)

        # Remove as linhas cobertas pela coluna escolhida
        linhas_nao_cobertas -= set(dados.colunas[melhor_coluna].linhascobertas)

    # Garante que a solução é válida
    assert valid_solution(solucao, dados)

    custo = sum([dados.colunas[j].custo for j in solucao])

    return solucao, custo

# Função que implementa o algoritmo de melhoramento para o Set Covering Problem
# Recebe como parâmetro uma solução dada pelo construtivo guloso e os dados do problema
# Retorna uma solução melhorada
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


# Função que executa o primeiro algoritmo construtivo guloso com o algoritmo de melhoramento
# O algoritmo é executado 20000 vezes e a cada iteração, se a solução encontrada for melhor que a anterior,
# a solução é atualizada
# Ao final, é retornada a melhor solução encontrada
def construtivo_com_melhoramento(dados, iteracoes):

    inicio = time.time()

    solucao1, custo = construtivo(dados)
    solucao = (solucao1, custo)

    for _ in range(iteracoes):
        solucao = melhoramento(solucao, dados)
        if solucao[1] < custo:
            custo = solucao[1]
            solucao1 = deepcopy(solucao[0])
            solucao_ajustada = [c + 1 for c in solucao1]
            print(f"Solucao encontrada: {solucao_ajustada}, {custo}")

    solucao_ajustada_final = [c + 1 for c in solucao1]
    fim = time.time()

    print(f"A melhor solucao encontrada foi: {solucao_ajustada_final}, com custo {custo}")
    print(f"Tempo de execucao para o Best Improvement: {fim - inicio} segundos")


# Função que executa o segundo algoritmo construtivo guloso com o algoritmo de melhoramento
# O algoritmo é executado 20000 vezes e a cada iteração, se a solução encontrada for melhor que a anterior,
# a solução é atualizada
# Ao final, é retornada a melhor solução encontrada
def construtivo_2_com_melhoramento(dados, iteracoes):

    inicio = time.time()

    solucao1, custo = construtivo2(dados)
    solucao = (solucao1, custo)

    for _ in range(iteracoes):
        solucao = melhoramento(solucao, dados)
        if solucao[1] < custo:
            custo = solucao[1]
            solucao_ajustada = [c + 1 for c in solucao1]
            print(f"Solucao encontrada: {solucao_ajustada}, {custo}")

    solucao_ajustada_final = [c + 1 for c in solucao1]
    fim = time.time()

    print(f"A melhor solucao encontrada foi: {solucao_ajustada_final}, com custo {custo}")
    print(f"Tempo de execucao para o first Improvement : {fim - inicio} segundos")


# Função que verifica se a solução encontrada é válida, ou seja,
# se a escolha das colunas da solução cobre todas as linhas
def valid_solution(solucao, dados):
    rows = [0] * dados.nlinhas

    for c in solucao:
        for r in dados.colunas[c].linhascobertas:
            rows[r - 1] = 1

    return sum(rows) == dados.nlinhas


def main():

    nome_do_arquivo = sys.argv[1]
    num_iteracoes = int(sys.argv[2])
    print("Set Covering Problem - Heuristica Construtiva 1\n")
    construtivo_com_melhoramento(ler_arquivo(nome_do_arquivo), num_iteracoes)
    print("\nSet Covering Problem - Heuristica Construtiva 2\n")
    construtivo_2_com_melhoramento(ler_arquivo(nome_do_arquivo), num_iteracoes)

    # print("construtivo 1\n")
    # for _ in range(20):
    #     solucao, custo = construtivo(ler_arquivo(nome_do_arquivo))
    #     solucao_ajustada = [c + 1 for c in solucao]
    #     print(f"Solucao encontrada: {solucao_ajustada}, com custo {custo}\n")

    # print("construtivo 2\n")
    # for _ in range(20):
    #     solucao, custo = construtivo2(ler_arquivo(nome_do_arquivo))
    #     solucao_ajustada = [c + 1 for c in solucao]
    #     print(f"Solucao encontrada: {solucao_ajustada}, com custo {custo}\n")
    
if __name__ == "__main__":
    main()
