import handle_car_dataset
import handle_congressional_dataset
import handle_leaf_dataset
import classificadores

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

  if entrada == 1:
    base_treino, base_teste, classificacoes_treino, classificacoes_teste = handle_car_dataset.get()
    classificadores.exec(
      base_treino, 
      base_teste, 
      classificacoes_treino, 
      classificacoes_teste, 
      title_base=labels_base[entrada-1]
    )

  if entrada == 2:
    base_treino, base_teste, classificacoes_treino, classificacoes_teste = handle_congressional_dataset.get()
    classificadores.exec(
      base_treino, 
      base_teste, 
      classificacoes_treino, 
      classificacoes_teste, 
      title_base=labels_base[entrada-1]
    )

  if entrada == 3:
    base_treino, base_teste, classificacoes_treino, classificacoes_teste = handle_leaf_dataset.get(mapear=False)
    classificadores.exec(
      base_treino, 
      base_teste, 
      classificacoes_treino, 
      classificacoes_teste,
      title_base=labels_base[entrada-1]
    )