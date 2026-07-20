import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


class RateCoder:

    def __init__(self, time_steps=50):
        self.time_steps = time_steps

    def encode_image(self, image):

        image = image.numpy()

        h, w = image.shape

        spikes = np.zeros((self.time_steps, h, w))

        for t in range(self.time_steps):

            random_matrix = np.random.rand(h, w)

            spikes[t] = random_matrix < image

        return spikes.astype(int)


# -----------------------
# Load MNIST
# -----------------------

transform = transforms.ToTensor()

dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

image, label = dataset[0]

image = image.squeeze()

print("Digit:", label)
print("Image Shape:", image.shape)

encoder = RateCoder(time_steps=50)

spike_tensor = encoder.encode_image(image)

print("Spike Tensor Shape:", spike_tensor.shape)

# -----------------------
# Visualization
# -----------------------

plt.figure(figsize=(12,6))

plt.subplot(121)

plt.imshow(image, cmap='gray')

plt.title(f"MNIST Digit : {label}")

plt.axis("off")

plt.subplot(122)

pixel_row = 14
pixel_col = 14

plt.step(
    range(50),
    spike_tensor[:, pixel_row, pixel_col],
    where='mid'
)

plt.ylim(-0.2,1.2)

plt.xlabel("Time")

plt.ylabel("Spike")

plt.title(f"Spike Train of Pixel ({pixel_row},{pixel_col})")

plt.tight_layout()

plt.show()