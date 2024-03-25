import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

epsilon = 0.1
nodes = 20
x_length = 50
y_length = 50

desiredDistance = 15
scalingFactor = 1.2
interactionRange = (desiredDistance * scalingFactor)



def euclidean_norm(x1, x2, y1, y2):
    return np.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def euclidean_norm2(x1, x2, y1, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1) **2)

#Sigma norm
def sigma_norm(z):
    return (np.sqrt(1 + (epsilon * (z ** 2))) - 1) / epsilon

x_coordinates = np.random.uniform(0, x_length, nodes)
y_coordinates = np.random.uniform(0, y_length, nodes)

i = 5
j = 8
x1 = 2
y1 = 2
x2 = -2
y2 = -2

N = 30
M = 2
X = 50
# testEuclideanNorm = euclidean_norm(x_coordinates[i], x_coordinates[j], y_coordinates[i], y_coordinates[j]) 
testEuclideanNorm = euclidean_norm(x1, x2, y1, y2)
print(testEuclideanNorm)
# t2 = np.linalg.norm([x2, y2] - [x1, y1])


nodes = (np.random.rand(N, M) * X)
# print (nodes[:, 0])
print (nodes[9])
# print(t2)
print (nodes[1] - nodes[2])
print (np.linalg.norm(nodes[1] - nodes[2]))
print(euclidean_norm(nodes[1, 0], nodes[2, 0], nodes[1, 1], nodes[2,1]))

print(interactionRange)
print(sigma_norm(interactionRange))