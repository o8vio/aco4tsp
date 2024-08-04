import numpy as np
import matplotlib.pyplot as plt
from main import *


points = np.random.uniform(low=-100, high=100, size=(100,2))

nntour = nearest_neighbour_heuristic(distances_matrix(points), rm.randint(0, np.shape(points)[0] - 1))
acotour = acs4tsp(points, number_of_ants=15, q0=0.9, k_nearest=15, alpha=0.1, beta=2, number_of_iterations=200)

# plot tour lengths obtained with each method
print('nn tour: ',tour_length(distances_matrix(points), nntour))
print('aco tour: ',tour_length(distances_matrix(points), acotour[-1]))


# plot the actual tours
fig, axs = plt.subplots(1, 2, figsize=(12,6))
plt.subplots_adjust(bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
axs[0].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
axs[0].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
data1 = {'x' : points[nntour,0], 'y' : points[nntour,1]}
data2 = {'x' : points[acotour[-1],0], 'y' : points[acotour[-1],1]}

axs[0].plot(np.append(data1['x'], np.array([points[nntour[0],0]])), np.append(data1['y'], np.array([points[nntour[0],1]])), c = 'firebrick')
axs[0].legend(['nntour'], loc='upper right')

axs[1].plot(np.append(data2['x'], np.array([points[acotour[-1][0],0]])), np.append(data2['y'], np.array([points[acotour[-1][0],1]])),c = 'olive')
axs[1].legend(['ACS tour'], loc='upper right')


# generate gif with tour evolution obtained via ant colony optimization
from matplotlib.animation import FuncAnimation
from IPython import display

def partial(nodes, tours):
    def anim(frame):
      data = {'x' : nodes[tours[frame],0], 'y' : nodes[tours[frame],1]}
      ax.clear()
      ax.scatter('x','y',data=data)
      ax.legend(['best #'+str(frame+1)], loc='upper right')
      return ax.plot(np.append(data['x'], np.array([nodes[tours[frame][0], 0]])), np.append(data['y'], np.array([nodes[tours[frame][0], 1]])))
    return anim

fig, ax = plt.subplots(figsize=(7,7))
animation = FuncAnimation(fig, partial(nodes=points, tours=acotour), frames=np.shape(acotour)[0], interval=200)
plt.close(fig)

# save gif
animation.save('evolution.gif')
