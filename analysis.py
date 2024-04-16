import cv2
import numpy as np
import matplotlib as plt


def calculate_velocity_and_acceleration(positions, timestamps):

    # Assume positions is a matrix of shape (players, time_steps, [x, y]) and timestamps is an array of length time_steps:
    velocities = np.zeros_like(positions)       # Create a velocities matrix with the same shape
    accelerations = np.zeros_like(positions)    # Create an accelerations matrix with the same shape

    # Per player per time step:
    for player in range(positions.shape[0]):
        for time_step in range(1, positions.shape[1]):
            delta_s = positions[player, time_step] - positions[player, time_step - 1]
            delta_t = timestamps[time_step] - timestamps[time_step - 1]
            velocities[player, time_step] = np.linalg.norm(delta_s) / delta_t

            if time_step > 1:
                delta_v = velocities[player, time_step] - velocities[player, time_step - 1]
                accelerations[player, time_step] = np.linalg.norm(delta_v) / delta_t

    return velocities, accelerations


def plot_players_stats(velocities, accelerations, timestamps):
    # Create a figure and a set of subplots for velocity and acceleration
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot velocities
    for player in range(velocities.shape[0]):
        axs[0].plot(timestamps[1:], velocities[player, 1:], label=f'Player {player+1}')
    axs[0].set_title('Velocity of Each Player Over Time')
    axs[0].set_ylabel('Velocity (m/s)')
    axs[0].legend(loc='upper right')
    
    # Plot accelerations
    for player in range(accelerations.shape[0]):
        axs[1].plot(timestamps[2:], accelerations[player, 2:], label=f'Player {player+1}')
    axs[1].set_title('Acceleration of Each Player Over Time')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Acceleration (m/s^2)')
    axs[1].legend(loc='upper right')
    
    # Display the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Example usage with dummy data:
    num_players = 11
    num_time_steps = 60

    # Creating pretent positions so that i can test my code:
    real_world_positions = np.random.rand(num_players, num_time_steps, 2) * 10  # dummy positions in meters
    timestamps = np.linspace(0, 10, num=num_time_steps)  # dummy timestamps in seconds

    # Calculate velocities and accelerations:
    velocities, accelerations = calculate_velocity_and_acceleration(real_world_positions, timestamps)

    # Visualize the results:
    plot_players_stats(velocities, accelerations, timestamps)
