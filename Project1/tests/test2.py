import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
Global Variables
"""
# Parameters for the moving point
start_position = (150, 150)  # Starting position
amplitude = 50
frequency = 0.2  # Cycles per unit time
phase_shift = np.pi / 2  # Starts at rest, at peak amplitude

# Time parameters
delta_t = 0.009
iterations = 600

# Initialize position and velocity logging for the moving point
logPointPosition_y = np.zeros(iterations, dtype=np.float16)
logPointVelocity_y = np.zeros(iterations, dtype=np.float16)

"""
Update Function for the Moving Point
"""
def update_point():
    for i in range(iterations):
        time = i * delta_t
        
        # Calculate the new y-coordinate based on a sine wave
        y_position = start_position[1] + amplitude * np.sin(2 * np.pi * frequency * time)
        
        # Calculate y-velocity based on the derivative of the sine wave
        y_velocity = amplitude * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * time)
        
        # Log position and velocity
        logPointPosition_y[i] = y_position
        logPointVelocity_y[i] = y_velocity

"""
Plotting Functions
"""
def plot_point_trajectory():
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(0, iterations * delta_t, delta_t), logPointPosition_y, label='Y Position')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Point Y Position Over Time')
    plt.legend()
    plt.show()

def plot_point_velocity():
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(0, iterations * delta_t, delta_t), logPointVelocity_y, label='Y Velocity', color='red')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Point Y Velocity Over Time')
    plt.legend()
    plt.show()

"""
Main
"""
update_point()  # Update the point's position and velocity over time
plot_point_trajectory()  # Plot the point's trajectory over time
plot_point_velocity()  # Plot the point's velocity over time