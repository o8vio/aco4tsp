from nntour import *
from aux import *


def acs4tsp(nodes, number_of_ants, q0, k_nearest, alpha, beta, number_of_iterations):
  """
  returns array with the evolution of the best tour found
  Parameters:
  M : number of ants to simulate in each iteration
  q0 : probability for choosing a method
  alpha, beta : pheromone updating rule parameters
  # T : number of iterations
  """

  number_of_nodes = np.shape(nodes)[0]
  dist_matrix = distances_matrix(nodes)
  k_nearest_matrix = k_nearest_neighbour_list_matrix(dist_matrix, k_nearest)

  # we initialize the algorithm with the nearest-neighbour heuristic solution
  best_evol = nearest_neighbour_heuristic(dist_matrix, rm.randint(0, number_of_nodes - 1))
  best_ever = best_evol

  # we initialize the pheromone values
  tau0 = 1 / (number_of_nodes * tour_length(dist_matrix, best_ever))
  pheromone_values = np.full((number_of_nodes, number_of_nodes), tau0)

  iter = 0
  stuck = 0

  while iter < number_of_iterations and stuck < 100: # we add a condition to break if converged to local minima

    best_iter = ant_tours(dist_matrix, k_nearest_matrix, number_of_ants, pheromone_values, q0, alpha, beta, tau0)

    stuck += 1

    if tour_length(dist_matrix, best_iter) < tour_length(dist_matrix, best_ever):
      best_ever = best_iter
      best_evol = np.vstack([best_evol, best_ever])

      stuck = 0

    # we apply global pheromone update rule:
    for i in range(number_of_nodes - 1):
      pheromone_values[best_ever[i], best_ever[i+1]] = (1-alpha)*pheromone_values[best_ever[i], best_ever[i+1]] + alpha/tour_length(dist_matrix, best_ever)
      pheromone_values[best_ever[i+1], best_ever[i]] = (1-alpha)*pheromone_values[best_ever[i+1], best_ever[i]] + alpha/tour_length(dist_matrix, best_ever)
    pheromone_values[best_ever[-1], best_ever[0]] = (1-alpha)*pheromone_values[best_ever[-1], best_ever[0]] + alpha/tour_length(dist_matrix, best_ever)
    pheromone_values[best_ever[0], best_ever[-1]] = (1-alpha)*pheromone_values[best_ever[0], best_ever[-1]] + alpha/tour_length(dist_matrix, best_ever)

    print('current iteration: '+str(iter+1)+'\n'+'current best-so-far tour length: '+str(tour_length(dist_matrix, best_ever)), 'stuck count: ', stuck)
    iter += 1

  print('\nIn '+str(number_of_iterations)+' iterations the algorithm updated the best ever solution '+str(np.shape(best_evol)[0])+' times.')

  return best_evol
