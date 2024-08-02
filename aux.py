def k_nearest_neighbour_list_matrix(dist_matrix, k):
  # returns matrix with every node's k nearest neighbours

  number_of_nodes = np.shape(dist_matrix)[0]
  matrix = np.zeros((number_of_nodes, k))

  i = 0
  while i < number_of_nodes:

    matrix[i] = np.sort(np.argsort(dist_matrix[i])[1:k+1])
    i += 1

  return matrix.astype(int)


def tour_length(dist_matrix, tour):

  number_of_nodes = np.shape(dist_matrix)[0]
  distances = np.append(np.array([dist_matrix[tour[i], tour[i+1]] for i in range(number_of_nodes - 1)]), np.array([dist_matrix[tour[-1], tour[0]]]))

  return np.sum(distances)



def choose_next_node(dist_matrix, k_nearest_matrix, current_node, visited_nodes, pheromone_values, q0, alpha, beta, tau0):

  number_of_nodes = np.shape(dist_matrix)[0]
  nearest_in_visited = np.array([n in visited_nodes for n in k_nearest_matrix[current_node]])

  if np.all(nearest_in_visited):
    candidates = np.delete(np.arange(number_of_nodes), visited_nodes)
  else:
    candidates = k_nearest_matrix[current_node][np.logical_not(nearest_in_visited)]

  candidates_values = pheromone_values[current_node, candidates] / (dist_matrix[current_node, candidates])**beta

  if rm.uniform(0,1) <= q0:
    choice = candidates[np.argmax(candidates_values)]
  else:
    probabilities = candidates_values / np.sum(candidates_values)
    choice = np.random.choice(candidates, p=probabilities)

  pheromone_values[current_node, choice] = (1-alpha)*pheromone_values[current_node, choice] + alpha*tau0
  pheromone_values[choice, current_node] = (1-alpha)*pheromone_values[choice, current_node] + alpha*tau0

  return choice


def ant_tours(dist_matrix, k_nearest_matrix, number_of_ants, pheromone_values, q0, alpha, beta, tau0):

  number_of_nodes = np.shape(dist_matrix)[0]

  starting_nodes = np.array([rm.randint(0, number_of_nodes - 1) for j in range(number_of_ants)])
  tours = np.array([[j] for j in starting_nodes])

  while np.shape(tours)[1] < number_of_nodes:

    next_nodes = np.array([[-1] for j in range(number_of_ants)])
    m = 0
    while m < number_of_ants:
      next_nodes[m][0] = choose_next_node(dist_matrix, k_nearest_matrix, tours[m][-1], tours[m], pheromone_values, q0, alpha, beta, tau0)
      m += 1

    tours = np.hstack((tours, next_nodes))

  m = 0
  while m < number_of_ants:
    pheromone_values[tours[m,-1], tours[m,0]] = (1-alpha)*pheromone_values[tours[m,-1], tours[m,0]] + alpha*tau0
    pheromone_values[tours[m,0], tours[m,-1]] = (1-alpha)*pheromone_values[tours[m,0], tours[m,-1]] + alpha*tau0
    m += 1

  return tours[np.argmin(np.array([tour_length(dist_matrix, tours[m]) for m in range(number_of_ants)]))]
