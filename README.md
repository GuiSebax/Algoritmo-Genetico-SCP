Para compilar o algoritmo, é necessário que o usuário tenho Python3 instalado em sua máquina

Para testar usando os testes que nós usamos para obter as respostas, basta rodar os seguintes comandos:

Para uma população de 100 e uma geração de 100 com busca local use

```.\run_tests100.bat```

E ele ira adicionar os resultados em um ```output100.txt```

Para uma população de 100 e uma geração de 500 com busca local use

```.\run_tests500.bat```
    
E ele ira adicionar os resultados em um ```output500.txt```
    
Agora caso queira sem busca local, basta rodar os seguintes comandos:

   ```.\run_tests100noLocalSearch.bat``` ou 
  ```.\run_tests500noLocalSearch.bat```


Agora se o usuário quiser rodar o algoritmo com os seus próprios parâmetros, basta rodar o seguinte comando:

```python AG.py <local do arquivo de entrada> <tamanho da população> <número de gerações> <probabilidade de mutação> <elitismo> <com busca local ou sem>```
    O ultimo parâmetro (com busca local ou sem) se quiser com buscal local basta colocar 1 e se não quiser basta colocar 0


Exemplo:
```python AG.py entradas/Teste_01.dat 100 100 0.9 0.1 1```



![image](https://github.com/GuiSebax/Algoritmo-Gen-tico---MOA/assets/103221587/2d5ec170-c9c0-4c2f-a3ec-835b43c54f2e)
