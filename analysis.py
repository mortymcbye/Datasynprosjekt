"""
1. We wish to retrieve information on following metrics for EACH object:
    - Velocity
    - Acceleration
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_velocity_and_acceleration(positions, timestamps):
    # Assume positions is a matrix of shape (players, time_steps, [x, y]) and timestamps is an array of length time_steps:
    velocities = np.zeros_like(positions[:, :, 0])  # Create a velocities matrix with the same shape
    accelerations = np.zeros_like(positions[:, :, 0])  # Create an accelerations matrix with the same shape

    # Per player per time step:
    for player in range(positions.shape[0]):
        for time_step in range(1, positions.shape[1]):
            delta_s = np.linalg.norm(positions[player, time_step] - positions[player, time_step - 1])
            delta_t = timestamps[time_step] - timestamps[time_step - 1]
            velocities[player, time_step] = delta_s / delta_t

            if time_step > 1:
                delta_v = velocities[player, time_step] - velocities[player, time_step - 1]
                accelerations[player, time_step] = delta_v / delta_t

    return velocities, accelerations



def plot_velocity(velocities, timestamps):
    # Create a figure for velocity plot
    plt.figure(figsize=(10, 5))
    
    # Define colors for Team 1 (red) and Team 2 (blue)
    team1_colors = sns.color_palette("Reds", velocities.shape[0] // 2)
    team2_colors = sns.color_palette("Blues", velocities.shape[0] // 2)
    
    # Plot velocities
    for player in range(velocities.shape[0]):
        if player < velocities.shape[0] // 2:  # Team 1
            color = team1_colors[player]
            label = f'Team 1, Player {player % (velocities.shape[0] // 2) + 1}'
        else:  # Team 2
            color = team2_colors[player % (velocities.shape[0] // 2)]
            label = f'Team 2, Player {player % (velocities.shape[0] // 2) + 1}'
        plt.plot(timestamps[1:], velocities[player, 1:], color=color, label=label)
    plt.title('Velocity of Each Player Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend(loc='upper right')
    
    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_acceleration(accelerations, timestamps):
    # Create a figure for acceleration plot
    plt.figure(figsize=(10, 5))
    
    # Define colors for Team 1 (red) and Team 2 (blue)
    team1_colors = sns.color_palette("Reds", accelerations.shape[0] // 2)
    team2_colors = sns.color_palette("Blues", accelerations.shape[0] // 2)
    
    # Plot accelerations
    for player in range(accelerations.shape[0]):
        if player < accelerations.shape[0] // 2:  # Team 1
            color = team1_colors[player]
            label = f'Team 1, Player {player % (accelerations.shape[0] // 2) + 1}'
        else:  # Team 2
            color = team2_colors[player % (accelerations.shape[0] // 2)]
            label = f'Team 2, Player {player % (accelerations.shape[0] // 2) + 1}'
        plt.plot(timestamps[2:], accelerations[player, 2:], color=color, label=label)
    plt.title('Acceleration of Each Player Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.legend(loc='upper right')
    
    # Display the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Example usage with dummy data:
    num_players = 22
    num_time_steps = 60

    # Creating pretend positions so that I can test my code:
    real_world_positions = np.random.rand(num_players, num_time_steps, 2) * 10  # dummy positions in meters
    timestamps = np.linspace(0, 10, num=num_time_steps)  # dummy timestamps in seconds

    # Calculate velocities and accelerations:
    velocities, accelerations = calculate_velocity_and_acceleration(real_world_positions, timestamps)

    # Visualize the results:
    plot_velocity(velocities, timestamps)
    plot_acceleration(accelerations, timestamps)
