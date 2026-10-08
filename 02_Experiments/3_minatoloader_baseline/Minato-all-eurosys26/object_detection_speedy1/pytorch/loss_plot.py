import torch
import matplotlib.pyplot as plt

# Load loss values
losses = torch.load("losses_speedy.pth")

# Add a high value at the beginning
losses = [6] + losses

# Plot
plt.figure(figsize=(10, 5))
plt.plot(losses, label="Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Function Fluctuation Over Training")
plt.legend()
plt.savefig("loss_plot.png")
plt.show()
