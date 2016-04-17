import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  x = low_dim_embs[:,0]
  y = low_dim_embs[:,1]
  plt.scatter(x, y, c = np.asarray(labels), cmap=plt.cm.gnuplot2)
  cb = plt.colorbar()
  loc = np.arange(0,9,9/9)
  cb.set_ticks(loc)
  cb.set_ticklabels(range(10))
  plt.savefig(filename)

low_dim_embs = np.loadtxt(sys.argv[1], delimiter = ',')
labels = np.loadtxt(sys.argv[2], delimiter = ',')
labels = map(lambda x: x.index(1), labels.tolist())
plot_with_labels(low_dim_embs, labels)

