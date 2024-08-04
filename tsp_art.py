import numpy as np
import random as rm
import urllib.request
from main import *
from PIL import Image, ImageOps


def image_points(image_array, k, gamma):
  # assumes k divides both image_array dimensions

  # we compute greyscale means in every cell 
  mu = np.zeros((int(np.shape(image_array)[0]/k), int(np.shape(image_array)[1]/k)))
  i = 0
  while i < np.shape(mu)[0]:
    j = 0
    while j < np.shape(mu)[1]:
      mu[i,j] = (1/(k**2))*np.sum(image_array[k*i:k*(i+1),k*j:k*(j+1)])
      j += 1
    i += 1

  # we compute the ammount of points to sample in every cell
  g = np.zeros((int(np.shape(image_array)[0]/k), int(np.shape(image_array)[1]/k)))
  i = 0
  while i < np.shape(mu)[0]:
    j = 0
    while j < np.shape(mu)[1]:
      if mu[i,j] <= 200:
        g[i,j] = int(gamma - np.floor((gamma*mu[i,j])/255))
      j += 1
    i += 1

  image_points = np.zeros((int(np.sum(g)), 2))

  counter = 0
  i = 0
  while i < np.shape(mu)[0]:
    j = 0
    while j < np.shape(mu)[1]:
      image_points[int(counter): int(counter + g[i,j]), 0] = np.random.uniform(100*(j/np.shape(mu)[1]), 100*((j+1)/np.shape(mu)[1]), int(g[i,j]))
      image_points[int(counter): int(counter + g[i,j]), 1] = np.random.uniform(((np.shape(mu)[0] - i - 1)/np.shape(mu)[0])*100, 100*((np.shape(mu)[0] - i)/np.shape(mu)[0]), int(g[i,j]))
      counter += int(g[i,j])
      j += 1
    i += 1

  return image_points



url = "https://i.postimg.cc/prYHdrQ7/banana.png"
urllib.request.urlretrieve(url, "banana.png")

banana = ImageOps.grayscale(Image.open(r'banana.png'))
banana_asarray = np.asarray(banana)
banana_asarray = np.vstack((banana_asarray, np.full((2,280), 255)))


banana_points = image_points(banana_asarray, k=5, gamma=4)

nntour_banana = nearest_neighbour_heuristic(distances_matrix(banana_points), rm.randint(0, np.shape(banana_points)[0] - 1))
acotour_banana = acs4tsp(banana_points, number_of_ants=15, q0=0.9, k_nearest=15, alpha=0.1, beta=2, number_of_iterations=500)

fig, axs = plt.subplots(1, 2, figsize=(14,5))
plt.subplots_adjust(bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
axs[0].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
axs[0].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
data1 = {'x' : banana_points[nntour_banana,0], 'y' : banana_points[nntour_banana,1]}
data2 = {'x' : banana_points[acotour_banana[-1],0], 'y' : banana_points[acotour_banana[-1],1]}

axs[0].plot(np.append(data1['x'], np.array([banana_points[nntour_banana[0],0]])), np.append(data1['y'], np.array([banana_points[nntour_banana[0],1]])), c = 'firebrick')
axs[0].legend(['nntour'], loc='upper right')

axs[1].plot(np.append(data2['x'], np.array([banana_points[acotour_banana[-1][0],0]])), np.append(data2['y'], np.array([banana_points[acotour_banana[-1][0],1]])),c = 'olive')
axs[1].legend(['ACS tour'], loc='upper right')
