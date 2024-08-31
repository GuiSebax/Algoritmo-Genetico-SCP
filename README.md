# Algoritmo Genético - Problema de Cobertura de Conjuntos

Dentro deste repositório está o trabalho desenvolvido durante a matéria de Modelagem e Otimização Algorítmica, também possui um relatório para ter uma análise mais concisa do problema:

## Algoritmo de Cobertura de Conjuntos (Set Covering Problem)

O trabalho consiste na pesquisa e implementação de um algoritmo aseado na meta-heurística Algoritmos Genéticos combinado com busca local para o problema de cobertura de conjunto (set covering problem). Implementado em Python.

O trabalho propoe a implementação de um algoritmo baseado em Algoritmos Genéticos para a resolução do problema supracitado.

Para compilar o algoritmo, é necessário que o usuário tenho Python3 instalado em sua máquina

Para testar usando os testes que nós usamos para obter as respostas, basta rodar os seguintes comandos:

Para uma população de 100 e uma geração de 100 com busca local use (É necessário que o usuário preste atenção nos diretórios onde os arquivos estão):

`.\run_tests100.bat`

E ele ira adicionar os resultados em um `output100.txt`

Para uma população de 100 e uma geração de 500 com busca local use

`.\run_tests500.bat`

E ele ira adicionar os resultados em um `output500.txt`

Agora caso queira sem busca local, basta rodar os seguintes comandos:

`.\run_tests100noLocalSearch.bat` ou
`.\run_tests500noLocalSearch.bat`

Agora se o usuário quiser rodar o algoritmo com os seus próprios parâmetros, basta rodar o seguinte comando:

`python AG.py <local do arquivo de entrada> <tamanho da população> <número de gerações> <probabilidade de mutação> <elitismo> <com busca local ou sem>`
O ultimo parâmetro (com busca local ou sem) se quiser com buscal local basta colocar 1 e se não quiser basta colocar 0

Exemplo:
`python AG.py entradas/Teste_01.dat 100 100 0.9 0.1 1`
