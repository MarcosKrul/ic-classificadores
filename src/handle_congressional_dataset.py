import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def get(mapear=True, test_size=0.7):
  base = pd.read_csv(
    filepath_or_buffer=os.path.join(os.path.dirname(__file__), '../tmp/dataset/congressional/house-votes-84.data'), 
    header=None, 
    delimiter=','
  )

  atributo_maior_qnt = []

  for i in range(1, 17):
    aux = base.groupby(i).count()
    yes_count = aux[0][1]
    no_count = aux[0][2]
    atributo_maior_qnt.append('y' if yes_count >= no_count else 'n')

  for index_i, linha in base.iterrows():
    for index_j in range(1, 17):
      if linha[index_j] == '?':
        base.iloc[index_i, index_j] = atributo_maior_qnt[index_j-1]

  base_atributos = base.iloc[:, 1:].values
  base_classificacoes = base.iloc[:, 0].values

  if mapear:
    _, n_colunas = np.shape(base_atributos)
    for i in range(n_colunas):
      le = LabelEncoder()
      base_atributos[:, i] = le.fit_transform(base_atributos[:, i])

  base_treino, base_teste, classificacoes_treino, classificacoes_teste = train_test_split(
    base_atributos,
    base_classificacoes,
    test_size=test_size,
    random_state=42  
  )

  print('===========================================')
  print('Informações base de dados Congressional')
  print(f'Dimensão base inteira: {np.shape(base_atributos)}')
  print(f'Dimensão base de treino: {np.shape(base_treino)}')
  print(f'Dimensão base de testes: {np.shape(base_teste)}')
  print('===========================================')

  return base_treino, base_teste, classificacoes_treino, classificacoes_teste