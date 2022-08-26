import os
import warnings
import plot_bar
from datetime import datetime
from prettytable import PrettyTable
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def exec(base_treino, base_teste, classificacoes_treino, classificacoes_teste, title_base):
  warnings.filterwarnings('ignore')

  models = []
  resultados = []

  models.append(('Árvore de Decisão', DecisionTreeClassifier(random_state=0)))
  models.append(('Naive bayes Gaussian', GaussianNB()))
  models.append(('Naive bayes Multinomial', MultinomialNB()))
  models.append(('Naive bayes Bernoulli', BernoulliNB()))

  for nome, modelo in models:
    modelo.fit(base_treino, classificacoes_treino)
    previsoes = modelo.predict(base_teste)
    acuracia = accuracy_score(classificacoes_teste, previsoes)
    resultados.append(acuracia)

    if nome == 'Naive bayes Gaussian':
      plot_bar.exec(
        x_label='Classes',
        y_label='Quantidade de registros',
        data=modelo.class_count_,
        labels=modelo.classes_,
        title=f'Balanceamento da base de dados {title_base}'
      )

    if nome == 'Árvore de Decisão':
      now = datetime.now()
      dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
      
      export_graphviz(
        modelo,
        class_names=[f'{x}' for x in modelo.classes_],
        filled=True, rounded=True, 
        out_file=os.path.join(os.path.dirname(__file__), f'../tmp/dot_files/tree_{dt_string}.dot')
      )
    
    print_tabela = True
    tabela = PrettyTable()
    matriz = confusion_matrix(classificacoes_teste, previsoes)
    tabela.title = 'Matriz de confusão'
    tabela.field_names = ['', *modelo.classes_]

    for i in range(0, len(matriz)):
      if len(modelo.classes_) == len(matriz[i]):
        row = [modelo.classes_[i]]
        tabela.add_row([*row, *matriz[i]])
      else:
        print_tabela = False
        break
    
    print('===================================================================================')
    print(nome)
    print(f'Quantidade de propriedades: {modelo.n_features_in_}')
    print(f'Acurácia do modelo: {(acuracia*100):,.2f}% ({acuracia})')
    print(classification_report(classificacoes_teste, previsoes))
    if print_tabela:
      print(tabela)
    print('===================================================================================')

  plot_bar.exec(
    x_label='Algoritmo',
    y_label='Acurácia (%)',
    data=resultados,
    labels=[x for x, _ in models],
    title=f'Comparação modelos de machine learning base {title_base}',
    porcentagem=True
  )

  return acuracia
