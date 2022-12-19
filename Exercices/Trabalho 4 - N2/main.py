import random
import math
from tqdm import tqdm
import numpy as np

x_lo = 0
x_up = 14
n_points = 1000

possibles_x = np.linspace(x_lo, x_up, n_points)



# ----- Algoritmo Genetico -----

# Parametros
ITERACOES = 1000
POPULACAO = 50
TORNEIO_PROB = 0.4
PROB_MUTACAO = 0.1

# ---------- MUTAÇÃO ----------
def mutacao( individuo):
  if random.random() < PROB_MUTACAO:
    individuo_mutado = []
    aux_1 = random.randint(0, len(possibles_x)-1)
    aux_2 = random.randint(0, len(possibles_x)-1)
    while(aux_1 == aux_2):
      aux_2 = random.randint(0, len(possibles_x)-1)
    if aux_2 < aux_1:
      aux_3 = aux_1
      aux_1 = aux_2
      aux_2 = aux_3
    for i in range(aux_1):
      individuo_mutado.append(individuo[i])
    for i in range(aux_2-1, aux_1-1, -1):
      individuo_mutado.append(individuo[i])
    for i in range(aux_2, len(individuo)):
      individuo_mutado.append(individuo[i])
    return individuo_mutado
  return individuo

# ---------- CROSSOVER ----------

def crossover( individuos, pais):
  aux_1 = random.randint(0, len(possibles_x)-1)
  aux_2 = random.randint(0, len(possibles_x)-1)
  if aux_2 < aux_1:
    aux_3 = aux_1
    aux_1 = aux_2
    aux_2 = aux_3
  list_pai = individuos[pais[0]][aux_1:aux_2]
  filho = []
  i = 0
  while len(filho) < aux_1:
    if individuos[pais[1]][i] not in list_pai:
      filho.append(individuos[pais[1]][i])
    i+=1
  filho = filho + list_pai
  i = 0
  while len(filho) < len(possibles_x):
    if individuos[pais[1]][i] not in filho:
      filho.append(individuos[pais[1]][i])
    i+=1
  return filho

# ---------- SELEÇÃO ----------

def selecao_pais(fitness):
  pais = []
  for i in range(POPULACAO):
    pais.append(torneio(fitness))
  return pais

def torneio(fitness):
  pais = []
  for k in range(0, 2):
    individuos_torneio = []
    for i in range(POPULACAO):
      aux = random.random()
      if aux < TORNEIO_PROB:
        individuos_torneio.append(i)
    while len(individuos_torneio) < 2:
      individuos_torneio.append(random.randint(0, POPULACAO-1))
    best = fitness[individuos_torneio[0]]
    best_ind = individuos_torneio[0]
    for i in range(len(individuos_torneio)):
      if fitness[individuos_torneio[i]] > best:
        best = fitness[individuos_torneio[i]]
        best_ind = individuos_torneio[i]
    pais.append(best_ind)

  return pais



def gerar_solucao_aleatoria():
  return random.choice(possibles_x)


def calcular_fitness(x):
  fx = (x[0] * math.sin(x[0])) + (x[0] * math.cos(x[0]))
  return abs(fx)


# inicializando os individuos
individuos = []
fitness = []
fitness_tempo = []
for i in range(POPULACAO):
  individuos.append(gerar_solucao_aleatoria())
  fitness.append(calcular_fitness(individuos[-1]))

best = math.inf
best_route = []
for i in tqdm(range(ITERACOES)):
  pais = selecao_pais(fitness)
  filhos = []
  for j in pais:
    filhos.append(crossover(individuos ,j))
  individuos = filhos
  for k in range(POPULACAO):
    individuos[k] = mutacao(individuos[k])
  fitness = []
  for k in range(POPULACAO):
    fitness.append(calcular_fitness(individuos[k]))
    if fitness[-1] < best:
      best = fitness[-1]
      best_route = individuos[k]
  fitness_tempo.append(1/best)

print(best)

print(best_route)