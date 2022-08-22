import matplotlib.pyplot as plt

def exec(x_label, y_label, title, labels, data, porcentagem=False):
  plt.bar(labels, data, color='green')
  plt.xticks(labels)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title)
  if porcentagem:
    plt.gca().set_ylim([0, 1])
  plt.show()