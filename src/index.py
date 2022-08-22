import handle_car_dataset
import handle_congressional_dataset
import handle_leaf_dataset
import arvore_decisao
import naive_bayes
import plot_bar

if __name__ == '__main__':

  labels_base = ['Car Evaluation', 'Congressional', 'Leaf']

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
    a1 = naive_bayes.exec(
      base_treino, 
      base_teste, 
      classificacoes_treino, 
      classificacoes_teste, 
      title_base=labels_base[entrada-1]
    )
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
    a1 = naive_bayes.exec(
      base_treino, 
      base_teste, 
      classificacoes_treino, 
      classificacoes_teste, 
      title_base=labels_base[entrada-1]
    )
    a2 = arvore_decisao.exec(
      base_treino, 
      base_teste, 
      classificacoes_treino, 
      classificacoes_teste, 
      class_names=['democrat', 'republican'],
      feature_names=[
        'handicapped-infants', 'water-project-cost-sharing',
        'adoption-of-the-budget-resolution',
        'physician-fee-freeze', 'el-salvador-aid',
        'religious-groups-in-schools',
        'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
        'mx-missile','immigration', 'synfuels-corporation-cutback',
        'education-spending', 'superfund-right-to-sue','crime',
        'duty-free-exports', 'export-administration-act-south-africa'
      ]
    )

    resultados.append(a1)
    resultados.append(a2)

  if entrada == 3:
    base_treino, base_teste, classificacoes_treino, classificacoes_teste = handle_leaf_dataset.get()
    a1 = naive_bayes.exec(
      base_treino, 
      base_teste, 
      classificacoes_treino, 
      classificacoes_teste,
      title_base=labels_base[entrada-1]
    )
    a2 = arvore_decisao.exec(
      base_treino, 
      base_teste, 
      classificacoes_treino, 
      classificacoes_teste, 
      feature_names=[
        'Specimen Number', 'Eccentricity',
        'Aspect Ratio', 'Elongation',
        'Solidity', 'Stochastic Convexity',
        'Isoperimetric Factor', 'Maximal Indentation Depth',
        'Lobedness', 'Average Intensity',
        'Average Contrast', 'Smoothness',
        'Third moment', 'Uniformity', 'Entropy',
      ]
    )

    resultados.append(a1)
    resultados.append(a2)

  plot_bar.exec(
    x_label='Algoritmo',
    y_label='Acurácia (%)',
    data=resultados,
    labels=['Naive Bayes', 'Árvore de Decisão'],
    title=f'Comparação modelos de machine learning base {labels_base[entrada-1]}',
    porcentagem=True
  )