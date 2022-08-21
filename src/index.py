import handle_car_dataset
import handle_congressional_dataset
import arvore_decisao
import naive_bayes
import matplotlib.pyplot as plt

if __name__ == '__main__':

  print('[1] - Car evaluation')
  print('[2] - Congressional')
  print('[3] - Leaf')
  print('Escolha a base de dados: ')
  entrada = int(input())
  
  if entrada < 1 or entrada > 3:
    print('Entrada inválida')
    exit()

  resultados = []

  if entrada == 1:
    base_treino, base_teste, classificacoes_treino, classificacoes_teste = handle_car_dataset.get()
    a1 = naive_bayes.exec(base_treino, base_teste, classificacoes_treino, classificacoes_teste)
    a2 = arvore_decisao.exec(
      base_treino, 
      base_teste, 
      classificacoes_treino, 
      classificacoes_teste, 
      class_names=['unacc', 'acc', 'good', 'v-good'],
      feature_names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    )

    resultados.append(a1)
    resultados.append(a2)

  if entrada == 2:
    base_treino, base_teste, classificacoes_treino, classificacoes_teste = handle_congressional_dataset.get()
    a1 = naive_bayes.exec(base_treino, base_teste, classificacoes_treino, classificacoes_teste)
    a2 = arvore_decisao.exec(
      base_treino, 
      base_teste, 
      classificacoes_treino, 
      classificacoes_teste, 
      class_names=['democrat', 'republican'],
      feature_names=[
        'handicapped-infants',
        'water-project-cost-sharing',
        'adoption-of-the-budget-resolution',
        'physician-fee-freeze',
        'el-salvador-aid',
        'religious-groups-in-schools',
        'anti-satellite-test-ban',
        'aid-to-nicaraguan-contras',
        'mx-missile',
        'immigration',
        'synfuels-corporation-cutback',
        'education-spending',
        'superfund-right-to-sue',
        'crime',
        'duty-free-exports',
        'export-administration-act-south-africa'
      ]
    )

    resultados.append(a1)
    resultados.append(a2)

  labels = ['Naive Bayes', 'Árvore de Decisão']
  plt.bar(labels, resultados, color='red')
  plt.xticks(labels)
  plt.xlabel('Algoritmo')
  plt.ylabel('Acurácia (%)')
  plt.title('Comparação modelos de machine learning')
  plt.show()