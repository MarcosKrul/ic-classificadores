import warnings
import plot_bar
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from prettytable import PrettyTable

def exec(base_treino, base_teste, classificacoes_treino, classificacoes_teste, title_base):
  warnings.filterwarnings('ignore')

  naive_bayes = GaussianNB()
  naive_bayes.fit(base_treino, classificacoes_treino)
  previsoes = naive_bayes.predict(base_teste)
  acuracia = accuracy_score(classificacoes_teste, previsoes)

  plot_bar.exec(
    x_label='Classes',
    y_label='Quantidade de registros',
    data=naive_bayes.class_count_,
    labels=naive_bayes.classes_,
    title=f'Balanceamento da base de dados de treino {title_base}'
  )

  tabela = PrettyTable()
  matriz = confusion_matrix(classificacoes_teste, previsoes)
  tabela.title = 'Matriz de confusão'
  tabela.field_names = ['', *naive_bayes.classes_]

  for i in range(0, len(matriz)):
    row = [naive_bayes.classes_[i]]
    tabela.add_row([*row, *matriz[i]])

  print('===================================================================================')
  print('Informações do Naive Bayes')
  print(f'Quantidade de propriedades: {naive_bayes.n_features_in_}')
  print(f'Classes: {naive_bayes.classes_}')
  print(f'Quantidade de registros por classe: {naive_bayes.class_count_}')
  print(f'Percentual de registros por classe: {naive_bayes.class_prior_}')
  print(f'Acurácia do modelo: {(acuracia*100):,.2f}% ({acuracia})')
  print(classification_report(classificacoes_teste, previsoes))
  print(tabela)
  print('===================================================================================')

  return acuracia
