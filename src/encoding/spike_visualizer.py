import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------
# Rate Coder
# ---------------------------------------------------

class RateCoder:

    def __init__(self, time_steps=50):
        self.time_steps = time_steps

    def encode(self, image):

        image = image.numpy()

        h, w = image.shape

        spike_tensor = np.zeros((self.time_steps, h, w), dtype=int)

        for t in range(self.time_steps):

            random_matrix = np.random.rand(h, w)

            spike_tensor[t] = random_matrix < image

        return spike_tensor


# ---------------------------------------------------
# Load MNIST
# ---------------------------------------------------

transform = transforms.ToTensor()

dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

image, label = dataset[0]

image = image.squeeze()

encoder = RateCoder(time_steps=50)

spikes = encoder.encode(image)

# ---------------------------------------------------
# Average Spike Activity
# ---------------------------------------------------

average_activity = spikes.mean(axis=0)

# ---------------------------------------------------
# Spike Count
# ---------------------------------------------------

spike_counts = spikes.sum(axis=0)

# ---------------------------------------------------
# Raster Plot Data
# ---------------------------------------------------

rows = []
times = []

for r in range(28):
    for c in range(28):

        neuron_id = r * 28 + c

        spike_times = np.where(spikes[:, r, c] == 1)[0]

        for t in spike_times:
            rows.append(neuron_id)
            times.append(t)

# ---------------------------------------------------
# Visualization
# ---------------------------------------------------

fig = plt.figure(figsize=(14,10))

# Original Image
plt.subplot(221)
plt.imshow(image, cmap="gray")
plt.title(f"Original MNIST Digit : {label}")
plt.axis("off")

# Spike Train
plt.subplot(222)

row = 14
col = 14

plt.step(
    range(50),
    spikes[:, row, col],
    where="mid"
)

plt.title(f"Spike Train of Pixel ({row},{col})")
plt.xlabel("Time Step")
plt.ylabel("Spike")
plt.ylim(-0.2,1.2)

# Raster Plot
plt.subplot(223)

plt.scatter(
    times,
    rows,
    s=2
)

plt.title("Spike Raster Plot")
plt.xlabel("Time Step")
plt.ylabel("Neuron Index")

# Average Activity Heatmap
plt.subplot(224)

plt.imshow(
    average_activity,
    cmap="hot"
)

plt.title("Average Spike Activity")

plt.colorbar()

plt.tight_layout()

plt.savefig("outputs/spike_visualizer.png", dpi=300)

plt.show()

print("\nVisualization saved to outputs/spike_visualizer.png")