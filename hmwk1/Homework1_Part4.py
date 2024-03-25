import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


#calculate magnitude of point1 and point2
def distance_calc(x1, x2, y1, y2):
    return np.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

#return neighbors of a given node from adjacency matrix
def getNeighbors(matrix, node):
    return (np.nonzero(matrix[node])[0] + 1)


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

#adjacency matrix of nodes x nodes
adjacencyMatrix = np.zeros((nodes, nodes), dtype=int)

#set up matplotlib plt
fig, ax = plt.subplots()
ax.set_xlim(0, x_length)
ax.set_ylim(0, y_length)


#Add data from random coordinates
for x, y in zip(x_coordinates, y_coordinates):
    ax.plot(x, y, '^', color='purple')


#iterate through all nodes and calculate distance, if i node is within r to j node
#add the pair to the adjacency matrix.
for i in range(0, nodes):
    for j in range(i, nodes):
        mag = distance_calc(x_coordinates[i], x_coordinates[j], y_coordinates[i], y_coordinates[j])
        if mag <= r:
            adjacencyMatrix[i][j] = 1
            adjacencyMatrix[j][i] = 1

            #add line to plot
            if j != i:
                ax.plot([x_coordinates[i], x_coordinates[j]], [y_coordinates[i], y_coordinates[j]], 'b-') 
    adjacencyMatrix[i][i] = 0

    #Prints a list of (node, [neighbors])
    print (i + 1, getNeighbors(adjacencyMatrix, i))



    

#generate grid
plt.grid(True)
plt.show()