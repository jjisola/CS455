import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

"""
Global Variables
"""
#Parameters
nodes = 100
x_length = 150
y_length = 150
desiredDistance = 15
scalingFactor = 1.2
interactionRange = (desiredDistance * scalingFactor)

epsilon = 0.1
bump_h = 0.1
delta_t = 0.009

iterations = 540
capture_iteration = iterations / 6

a = 3
b = 3
c = np.abs(a-b)/np.sqrt(4 * a * b)

amplitude = 100
frequency = 0.025
phaseShift = 0

c1_alpha = 100
c2_alpha = 2 * np.sqrt(c1_alpha)

c1_gamma = 70
c2_gamma = 2 * np.sqrt(c1_gamma)

#global numpy arrays -> collect data
x_coordinates = np.random.uniform(0, x_length, nodes)
y_coordinates = np.random.uniform(0, y_length, nodes)

adjacencyMatrix = np.zeros((nodes, nodes), dtype=int)

velocity_x = np.zeros((nodes), dtype=np.float64)
velocity_y = np.zeros((nodes), dtype=np.float64)

logPositions_x = np.zeros((nodes, iterations), dtype=np.float32)
logPositions_y = np.zeros((nodes, iterations), dtype=np.float32)

logVelocitity_x = np.zeros((nodes, iterations), dtype=np.float32)
logVelocitity_y = np.zeros((nodes, iterations), dtype=np.float32)

logVelocitityMagnitude = np.zeros((nodes, iterations), dtype=np.float32)

logConnectivity = np.zeros((iterations), dtype=np.float32)



gammaStart = np.array((50, 50), dtype=np.float32)
dynamicRendezvous = np.array((50 + amplitude, 50), dtype=np.float32)
dynamicRendezvousVelocity = np.array((0, 0), dtype=np.float32)

logGamma_x = np.zeros((iterations), dtype=np.float32)
logGamma_y = np.zeros((iterations), dtype=np.float32)
logGamma_velocityMagnitude = np.zeros((iterations), dtype=np.float32)

centerOfMass_x = np.zeros((iterations), dtype=np.float32)
centerOfMass_y = np.zeros((iterations), dtype=np.float32)


""" 
Math Functions as described in "Flocking for Multi-Agent Dynamic Systems:
Algorithms and Theory"
"""
#calculate magnitude of point1 and point2
def euclidean_norm(x1, x2, y1, y2):
    return np.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

#Sigma norm
def sigma_norm(z):
    return (np.sqrt(1 + (epsilon * (z ** 2))) - 1) / epsilon


def sigma_1_gradient(z):
    return z / (np.sqrt(1 + (z**2)))

#Bump function
#bump_h is h (defined in parameters)
def bump_function(z):
    if z < bump_h and z >= 0:
        return 1
    elif z >= bump_h and z <= 1:
        temp = (z - bump_h) / (1 - bump_h)
        return ((1/2) * (1 + np.cos(np.pi * (temp))))
    else:
        return 0

#uneven sigmoidal
def phi(z):
    return (1/2) * (((a + b) * sigma_1_gradient(z + c)) + (a - b))

#Pairwise Potential action function -> vanish for z >= sigma_norm of interaction range
def phi_action(z):
    norm_r = sigma_norm(interactionRange)
    norm_d = sigma_norm(desiredDistance)
    return ((bump_function(z / norm_r)) * (phi(z - norm_d)) )

#spatial adjacency matrix of i and j in terms of q (position)
#returns list
def a_i_j(i, j):
    norm = sigma_norm(euclidean_norm(x_coordinates[i], x_coordinates[j], y_coordinates[i], y_coordinates[j]))
    norm_r = sigma_norm(interactionRange)
    return bump_function(norm / norm_r)
    
#sigma norm gradient of position i an dj
def n_i_j(i, j):
    diff = [x_coordinates[j] - x_coordinates[i], y_coordinates[j] - y_coordinates[i]]
    # euclidean_norm = euclidean_norm(x_coordinates[i], x_coordinates[j], y_coordinates[i], y_coordinates[j])
    denom = np.sqrt(1 + (epsilon * (euclidean_norm(x_coordinates[i], x_coordinates[j], y_coordinates[i], y_coordinates[j]))**2))
    return (diff) / (denom)

def getNeighbors(node):
    return (np.nonzero(adjacencyMatrix[node])[0])


def u_i_alpha(node):
    gradientSum = np.array([0, 0], dtype=np.float32)
    consensusSum = np.array([0, 0], dtype=np.float32)
    neighbors = getNeighbors(node)
    # print(neighbors)
    for neighbor in neighbors:
        v_1 = phi_action(sigma_norm(euclidean_norm(x_coordinates[node], x_coordinates[neighbor], y_coordinates[node], y_coordinates[neighbor])))
        gradientSum += n_i_j(node, neighbor) * v_1
        
        spat = a_i_j(node, neighbor)
        diff_x =np.array([velocity_x[neighbor] - velocity_x[node], velocity_y[neighbor] - velocity_y[node]], dtype=np.float32)
        consensusSum += (diff_x * spat)
    return (c1_alpha *gradientSum) + (c2_alpha*consensusSum)

def u_i_gamma(node, v_y, v_x):
    pDiff = np.array([0, 0], dtype=np.float32)
    vDiff = np.array([0, 0], dtype=np.float32)

    pDiff[0] = x_coordinates[node] - dynamicRendezvous[0]
    pDiff[1] = y_coordinates[node] - dynamicRendezvous[1]
    
    vDiff[0] = velocity_x[node] - v_x
    vDiff[1] = velocity_y[node] - v_y
    return -1 *(c1_gamma) * pDiff -  c2_gamma * vDiff

def u_i_(node, v_y, v_x):
    return u_i_alpha(node) + u_i_gamma(node, v_y, v_x)

#derivative of sin function
def gammaVelocity_y(chain, dt, multiplier, amp):
    return amp * chain * multiplier * np.cos(chain * dt + phaseShift)

def gammaVelocity_x(chain, dt, multiplier ,amp):
    return amp * chain * multiplier *(-1) * np.sin(chain * dt + phaseShift)


"""
Update Vectors
"""
#plots points based on x_coordinates and y_coordinates
#Also updates adjacency matrix
def plot_points(plot, iteration):
    #Add data from  coordinates
    if (plot == True):
            plt.plot(logGamma_x[0:iteration], logGamma_y[0:iteration])
            plt.plot(dynamicRendezvous[0], dynamicRendezvous[1], '^', color='purple')
            for x, y in zip(x_coordinates, y_coordinates):
                plt.plot(x, y, 'ro')

    # adjacencyMatrix = np.zeros((nodes, nodes), dtype=int)
    adjacencyMatrix.fill(0)
    for i in range(0, nodes):
        for j in range(i, nodes):
            mag = euclidean_norm(x_coordinates[i], x_coordinates[j], y_coordinates[i], y_coordinates[j])
            if mag <= interactionRange:
                adjacencyMatrix[i][j] = 1
                adjacencyMatrix[j][i] = 1

                #add line to plot
                if ((j != i) and (plot == True)):
                    plt.plot([x_coordinates[i], x_coordinates[j]], [y_coordinates[i], y_coordinates[j]], 'b-') 
        adjacencyMatrix[i][i] = 0




#each iteration = 1 * delta_t
def update_kinematics():
    time = 0
    for i in range(0, iterations):
        
        sumMass = np.array((0, 0) , dtype=np.float32)
        #Case: first iteration -> don't update anything
        if i == 0 : 
            plot_points(True, i)
            plt.title(f'Node Position at {i * delta_t} seconds')
            plt.show()
            for j in range(nodes):
                
                logPositions_x[j, i] = x_coordinates[j]
                logPositions_y[j, i] = y_coordinates[j]

                logVelocitity_x[j, i] = 0
                logVelocitity_y[j, i] = 0
                logVelocitityMagnitude[j, i] = 0
                logConnectivity[i] = (1 / nodes) * np.linalg.matrix_rank(adjacencyMatrix)

                logGamma_x[i] = dynamicRendezvous[0]
                logGamma_y[i] = dynamicRendezvous[1]
                
                sumMass[0] += x_coordinates[j]
                sumMass[1] += y_coordinates[j]
        #case: each next iteration -> update velocity, position
        else :
            
            time += delta_t
            gammaV_y = gammaVelocity_y((2 * np.pi * frequency), (i/10) , (1/10), amplitude)
            gammaV_x = gammaVelocity_x((2 * np.pi * frequency), (i/10), (1/10) ,amplitude)


            dynamicRendezvous[0] = gammaStart[0] + amplitude * np.cos(2 * np.pi * frequency * (i/10)  + phaseShift)
            dynamicRendezvous[1] = gammaStart[1] + amplitude * np.sin(2 * np.pi * frequency * (i/10)  + phaseShift)

            logGamma_x[i] = dynamicRendezvous[0]
            logGamma_y[i] = dynamicRendezvous[1]
            for j in range(nodes):

                u = u_i_(j, gammaV_y, gammaV_x)
                tmpVelocity = [velocity_x[j], velocity_y[j]]
                tmpPosition = [x_coordinates[j], y_coordinates[j]]

                v = tmpVelocity + (u * delta_t)
                p = tmpPosition + (v * delta_t) + ((1/2) * (delta_t ** 2) * u)

                [x_coordinates[j], y_coordinates[j]] = p
                [velocity_x[j], velocity_y[j]] = v
                logVelocitityMagnitude[j, i] = euclidean_norm(0, v[0], 0, v[1])


                logPositions_x[j, i] = x_coordinates[j]
                logPositions_y[j, i] = y_coordinates[j]
                
                [logVelocitity_x[j, i],  logVelocitity_y[j, i]] = v
                logConnectivity[i] = (1 / nodes) * np.linalg.matrix_rank(adjacencyMatrix)

                sumMass[0] += x_coordinates[j]
                sumMass[1] += y_coordinates[j]

        centerOfMass_x[i] = sumMass[0] / nodes
        centerOfMass_y[i] = sumMass[1] / nodes
        if (i != 0 and i % capture_iteration == 0):
            plot_points(True, i)
            plt.title(f'Node Position at {i * delta_t} seconds')
            plt.show()
        elif (i != 0 and i % capture_iteration != 0):
            plot_points(False, i)

        

def plotTrajectory():
    for node in range(nodes):
        plt.plot(logPositions_x[node], logPositions_y[node])
    plt.title(f'Trajectory of Nodes over {iterations * delta_t} seconds')
    plt.show()

def plotVelocity():
    for node in range(nodes):
        plt.plot(logVelocitityMagnitude[node])
    plt.title(f'Velocity magnitude of Nodes over {iterations * delta_t} seconds')
    plt.show()
def plotConnectivity():
    plt.plot(logConnectivity)
    plt.title(f'Connectivity of Nodes over {iterations * delta_t} seconds')
    plt.show()
def plotCMass():
    plt.plot(centerOfMass_x, centerOfMass_y)
    plt.plot(logGamma_x, logGamma_y)
    plt.title(f'Center of Mass of Nodes over {iterations * delta_t} seconds')
    plt.show()

"""
Main 
"""
plt.grid(True)
update_kinematics()
plotTrajectory()
plotVelocity()
plotConnectivity()
plotCMass()
# print(logPositions_x)
# plt.show()