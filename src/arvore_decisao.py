import os
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_graphviz
from datetime import datetime


def exec(base_treino, base_teste, classificacoes_treino, classificacoes_teste, feature_names=None, class_names=None):
  warnings.filterwarnings('ignore')

  arvore_decisao = DecisionTreeClassifier(random_state=0)
  arvore_decisao.fit(base_treino, classificacoes_treino)
  previsoes = arvore_decisao.predict(base_teste)
  acuracia = accuracy_score(classificacoes_teste, previsoes)

  now = datetime.now()
  dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
  
  export_graphviz(
    arvore_decisao,
    feature_names=feature_names,
    class_names=class_names,
    filled=True, rounded=True, 
    out_file=os.path.join(os.path.dirname(__file__), f'../tmp/dot_files/tree_{dt_string}.dot')
  )

  print('===============================================================')
  print('Informações da Árvore de decisão')
  print(f'Acurácia do modelo: {(acuracia*100):,.2f}% ({acuracia})')
  print(classification_report(classificacoes_teste, previsoes))
  print('===============================================================')

  return acuracia

