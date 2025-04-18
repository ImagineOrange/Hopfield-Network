# EthanCrouse 2/15/22
# Edited 4/18/2025 for sklearn MNIST, energy plot, resizing, style reversion, figsize, parameterized updates, comments

# imports
import numpy as np

# Use sklearn for MNIST data loading
from sklearn.datasets import fetch_openml

# for visualization
import matplotlib.pyplot as plt
import pygame
import time
import sys

"""
    Hopfield Network Object
    Implements a Hopfield network for pattern storage and retrieval (associative memory).
    Accepts input patterns as a matrix (samples x features).
"""
class Hopfield_Net:

    # Initialize network parameters and state
    def __init__(self, input):

        # Store the patterns to be memorized
        self.memory = np.array(input)
        print(f"\nSize of memory array: {self.memory.shape}\n")

        # Network dimensions and weights
        self.n = self.memory.shape[1] # Number of neurons (equal to feature count)
        self.weights = np.zeros((self.n, self.n)) # Weight matrix, initialized to zeros

        # --- Initialize network state with random bipolar values (-1, 1) ---
        self.state = np.random.choice([-1, 1], size=(self.n, 1)) # Current state of neurons
        self.i = self.state.copy() # Store the initial random state for later comparison

        # --- List to store energy history during simulation ---
        self.energy_history = [] # Stores energy values for plotting convergence


    # Train the network using the Hebbian learning rule
    def network_learning(self):
        """Constructs the weight matrix based on stored memory patterns."""
        # Calculate weights using the outer product sum of memory patterns
        self.weights = self.memory.T @ self.memory
        # Zero the diagonal of the weight matrix to prevent self-connections
        np.fill_diagonal(self.weights, 0)


    # Asynchronously update the state of 'num_updates' randomly chosen neurons
    def update_network_state(self, num_updates=1): # 'num_updates' controls how many neurons flip per call
        """Performs asynchronous updates on a specified number of neurons."""
        # Loop to update 'num_updates' randomly selected neurons
        for _ in range(num_updates):
            # Select a random neuron index
            self.rand_index = np.random.randint(0, self.n)
            # Calculate the activation for this neuron (weighted sum of inputs)
            # self.state is (n, 1), self.weights[idx, :] is (n,), result is scalar
            self.index_activation = np.dot(self.weights[self.rand_index, :], self.state)

            # Apply threshold rule (sign function) to update the neuron's state
            # Update state to -1 if activation is negative, +1 otherwise
            if self.index_activation < 0:
                self.state[self.rand_index, 0] = -1
            else:
                self.state[self.rand_index, 0] = 1

    # Calculate the Lyapunov energy function for the current network state
    def calculate_energy(self):
        """Calculates the Hopfield energy: E = -0.5 * S^T * W * S.
           Energy should decrease or stay constant as the network settles."""
        # Matrix multiplication for energy calculation
        energy = -0.5 * np.dot(self.state.T, np.dot(self.weights, self.state))
        # Result is a 1x1 matrix, return the scalar value
        return energy[0, 0]


# Main function to run the MNIST Hopfield simulation
def MNIST_Hopfield(updates_per_step=10): # Controls speed vs. granularity of visualization

    print(f"Running Hopfield simulation with {updates_per_step} neuron updates per step.")
    # --- Data Loading ---
    print("Fetching MNIST dataset using sklearn...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data # Image data (70000 x 784)
    print("MNIST data fetched.")
    print(f"Data shape: {X.shape}")

    # --- Data Preprocessing ---
    # Convert pixel values (0-255) to bipolar format (-1, 1) needed for Hopfield
    threshold = 20 # Threshold to binarize pixel values (chosen based on original code)
    X_binary = np.where(X > threshold, 1, -1)
    print(f"Data converted to bipolar format with threshold {threshold}.")

    # --- Memory Selection ---
    # Select a small number of random MNIST images to store as memories
    num_memories = 2 # Hopfield capacity is limited, typically store few patterns relative to N
    if num_memories > len(X_binary):
        num_memories = len(X_binary)

    # Randomly choose indices for the memory patterns
    memory_indices = np.random.choice(len(X_binary), num_memories, replace=False)
    memories_list = X_binary[memory_indices]
    print(f"Selected {num_memories} random patterns as memories.")

    # --- Network Initialization ---
    H_Net = Hopfield_Net(memories_list) # Create the network instance
    print("Training network (calculating weights)...")
    H_Net.network_learning() # Calculate the weight matrix
    print("Network training complete.")
    print("Initialized state with random bipolar values.") # Network starts from a random state

    # --- Pygame Visualization Setup ---
    cellsize = 23 # Size of each neuron ('pixel') in the display window
    img_dim = 28 # MNIST images are 28x28
    window_size = img_dim * cellsize # Calculate window dimensions

    pygame.init() # Initialize Pygame library
    surface = pygame.display.set_mode((window_size, window_size)) # Create display surface
    pygame.display.set_caption("Hopfield Network Recalling Pattern...") # Window title
    print("Pygame board initialized...")

    # --- Simulation Loop Control ---
    Running = True # Flag to keep the main loop running
    update_counter = 0 # Tracks total individual neuron updates performed

    # --- Energy Tracking Initialization ---
    initial_energy = H_Net.calculate_energy() # Calculate energy of initial random state
    H_Net.energy_history.append(initial_energy)
    # X-axis values for the energy plot (tracks update count when energy is recorded)
    energy_update_steps = [0]

    # --- Main Simulation Loop ---
    # Handles events, updates the network state, and redraws the display
    while Running:
        # --- Event Handling ---
        for event in pygame.event.get():
            # Handle window closing event
            if event.type == pygame.QUIT:
                Running = False # Set flag to exit the loop
                print("Exit event detected.")

                # --- Final Plotting (after simulation ends) ---
                print("Plotting final results...")
                # Create a 2x2 plot for memories and states
                fig, axes = plt.subplots(2, 2, figsize=(12, 7)) # Use specified figure size
                fig.suptitle('Hopfield Network: MNIST Pattern Retrieval', fontsize=16)

                # Plot Initial State (Top-Left)
                axes[0, 0].imshow(H_Net.i.reshape(img_dim, img_dim), cmap='gray', aspect='equal')
                axes[0, 0].set_title("Initial State")
                # Plot Final State (Top-Right)
                axes[0, 1].imshow(H_Net.state.reshape(img_dim, img_dim), cmap='gray', aspect='equal')
                axes[0, 1].set_title(f"Final State (after {update_counter} updates)") # Show total updates
                # Plot Memory 1 (Bottom-Left)
                axes[1, 0].imshow(H_Net.memory[0].reshape(img_dim, img_dim), cmap='gray', aspect='equal')
                axes[1, 0].set_title("Memory 1")
                # Plot Memory 2 (Bottom-Right), if it exists
                if H_Net.memory.shape[0] > 1:
                    axes[1, 1].imshow(H_Net.memory[1].reshape(img_dim, img_dim), cmap='gray', aspect='equal')
                    axes[1, 1].set_title("Memory 2")
                else:
                     axes[1, 1].set_title("No Memory 2")
                     axes[1, 1].axis('off') # Hide axes if no second memory

                # Configure plot appearance
                for ax in axes.flat:
                   ax.set_xticks([]) # Remove axis ticks
                   ax.set_yticks([])
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap
                # Show plot without blocking script execution (allows energy plot to show next)
                plt.show(block=False)

                # --- Plot Energy Evolution ---
                print("Plotting energy evolution...")
                plt.figure("Energy Evolution", figsize=(12, 7)) # Use specified figure size
                # Plot energy against the step number when it was calculated
                plt.plot(energy_update_steps, H_Net.energy_history)
                plt.title("Network Energy Over Updates")
                plt.xlabel(f"Neuron Update Step (Batches of {updates_per_step})") # Label x-axis clearly
                plt.ylabel("Energy")
                plt.grid(True) # Add grid for readability
                plt.show() # Show the energy plot (this will block until closed)

                # --- Cleanup ---
                pygame.quit() # Close the Pygame window
                print("Pygame board collapsed.")
                sys.exit() # Exit the script cleanly


        # --- Network Update Step ---
        # Update the state of multiple neurons in one go
        H_Net.update_network_state(num_updates=updates_per_step)
        # Increment total update counter by the number of updates performed in this step
        update_counter += updates_per_step

        # --- Energy Calculation Step ---
        # Calculate and store energy after the batch of updates
        current_energy = H_Net.calculate_energy()
        H_Net.energy_history.append(current_energy)
        # Store the total update count at which this energy was recorded for the plot x-axis
        energy_update_steps.append(update_counter)

        # --- Pygame Drawing Step ---
        # Get the current network state reshaped into a 2D grid for display
        cells = H_Net.state.reshape(img_dim, img_dim)

        # Fill the background
        surface.fill((50, 50, 50)) # Dark gray background
        # Draw each neuron's state as a colored rectangle
        for r in range(img_dim): # Iterate through rows
            for c in range(img_dim): # Iterate through columns
                # Map bipolar states (-1 black, 1 white) to Pygame colors
                if cells[r, c] == 1:
                     col = (255, 255, 255) # White for state 1
                else: # cells[r, c] == -1
                     col = (0, 0, 0) # Black for state -1

                # Draw the rectangle representing the neuron
                # (surface, color, (x_pos, y_pos, width, height))
                pygame.draw.rect(surface, col, (c * cellsize, r * cellsize,
                                                 cellsize - 1, cellsize - 1)) # -1 creates grid lines

        # Update window title with current update count
        pygame.display.set_caption(f"Hopfield Network Recalling... Updates: {update_counter}")
        # Refresh the display to show the newly drawn frame
        pygame.display.update()

        # Optional delay to slow down visualization (uncomment if needed)
        # time.sleep(0.01)


# --- Script Execution Entry Point ---
if __name__ == "__main__":
    # Run the simulation function with the default (or specified) updates per step
    # Example: MNIST_Hopfield(updates_per_step=100) to update 100 neurons at a time
    MNIST_Hopfield(updates_per_step=10)