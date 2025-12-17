# Original Author: Sebastian Raschka (https://github.com/rasbt/LLMs-from-scratch)
# Modified By: woodsj1206 (https://github.com/woodsj1206)
# Last Modified: 12/16/2025
import matplotlib.pyplot as plt
import os


def plot_data(epochs_seen, examples_seen, training_data, validation_data, dir_path, label="label"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, training_data, label=f"Training {label}")
    ax1.plot(epochs_seen, validation_data,
             linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    # Invisible plot for aligning ticks
    ax2.plot(examples_seen, training_data, alpha=0)
    ax2.set_xlabel("Examples Seen")

    fig.tight_layout()  # Adjust layout to make room

    save_path = os.path.join(dir_path, f"results-{label}.png")
    plt.savefig(save_path)
