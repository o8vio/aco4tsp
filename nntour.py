import numpy as np


def distances_matrix(nodes):
  
  number_of_nodes = np.shape(nodes)[0]
  matrix = np.zeros((number_of_nodes, number_of_nodes))
  i = 0
  while i < number_of_nodes:
    j = 0
    while j < number_of_nodes:
      matrix[i,j] = np.sum((nodes[i] - nodes[j])**2)**0.5
      j += 1
    i += 1

  return matrix


def nearest_neighbour_heuristic(dist_matrix, start):

  number_of_nodes = np.shape(dist_matrix)[0]
  nntour = np.array([start])

  while nntour.size < number_of_nodes:

    distances_to_non_visited = np.delete(dist_matrix[nntour[-1]], nntour)

    next_node = np.delete(np.arange(number_of_nodes), nntour)[np.argmin(distances_to_non_visited)]

    nntour = np.append(nntour, np.array([next_node]))

  return nntour
