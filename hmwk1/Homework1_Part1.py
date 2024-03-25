import numpy as np
import matplotlib.pyplot as plt


#parameters
nodes = 100
x_length = 150
y_length = 150
k = 1.2
d = 14
r = k * d

#set up coordinates
x_coordinates = np.random.uniform(0, x_length, nodes)
y_coordinates = np.random.uniform(0, y_length, nodes)

#set up matplotlib plt
fig, ax = plt.subplots()
ax.set_xlim(0, x_length)
ax.set_ylim(0, y_length)

#Add data from random coordinates
for x, y in zip(x_coordinates, y_coordinates):
    ax.plot(x, y, '^', color='purple')


#generate grid
plt.grid(True)
plt.show()

