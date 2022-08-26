import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def get(mapear=True, test_size=0.2):
  base = pd.read_csv(
    filepath_or_buffer=os.path.join(os.path.dirname(__file__), '../tmp/dataset/car_evaluation/car.data'), 
    header=None, 
    delimiter=','
  )

  base_atributos = base.iloc[:, 0:6].values
  base_classificacoes = base.iloc[:, 6].values

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
  print('Informações base de dados Car evaluation')
  print(f'Dimensão base inteira: {np.shape(base_atributos)}')
  print(f'Dimensão base de treino: {np.shape(base_treino)}')
  print(f'Dimensão base de testes: {np.shape(base_teste)}')
  print('===========================================')

  return base_treino, base_teste, classificacoes_treino, classificacoes_teste