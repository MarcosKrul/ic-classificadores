import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

def exec(base_treino, base_teste, classificacoes_treino, classificacoes_teste):
  warnings.filterwarnings('ignore')

  naive_bayes = GaussianNB()
  naive_bayes.fit(base_treino, classificacoes_treino)
  previsoes = naive_bayes.predict(base_teste)
  acuracia = accuracy_score(classificacoes_teste, previsoes)

  print('===================================================================================')
  print('Informações do Naive Bayes')
  print(f'Quantidade de propriedades: {naive_bayes.n_features_in_}')
  print(f'Quantidade de classes: {naive_bayes.classes_}')
  print(f'Quantidade de registros por classe: {naive_bayes.class_count_}')
  print(f'Percentual de registros por classe: {naive_bayes.class_prior_}')
  print(f'Acurácia do modelo: {(acuracia*100):,.2f}% ({acuracia})')
  print(classification_report(classificacoes_teste, previsoes))
  print('===================================================================================')

  return acuracia
