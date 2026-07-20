"""
NeuroSolver
---------------------------
Module: Rate Coding (Poisson Spike Encoding)

Author: Indrajith Berlin

Description:
Converts grayscale pixel intensities into spike trains.
Brighter pixels generate more spikes over time.

"""

import numpy as np
import matplotlib.pyplot as plt


class RateCoder:
    """
    Converts pixel intensities into spike trains using
    Poisson Rate Coding.
    """

    def __init__(self, time_steps=50):
        self.time_steps = time_steps

    def encode_pixel(self, pixel):

        probability = pixel / 255.0

        spikes = np.random.rand(self.time_steps) < probability

        return spikes.astype(int)

    def encode_image(self, image):

        h, w = image.shape

        spike_tensor = np.zeros(
            (self.time_steps, h, w),
            dtype=int
        )

        for i in range(h):
            for j in range(w):
                spike_tensor[:, i, j] = self.encode_pixel(image[i, j])

        return spike_tensor


def plot_results(image, spike_tensor, row, col):

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(image, cmap="gray")
    plt.scatter(col, row, color="red", s=80)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)

    plt.step(
        range(spike_tensor.shape[0]),
        spike_tensor[:, row, col],
        where='mid'
    )

    plt.ylim(-0.2,1.2)
    plt.xlabel("Time Step")
    plt.ylabel("Spike")
    plt.title(f"Spike Train ({row},{col})")

    plt.tight_layout()
    plt.show()


def main():

    image = np.array([
        [255,200,150],
        [100, 50, 25],
        [255,120, 0]
    ])

    encoder = RateCoder(time_steps=40)

    spike_tensor = encoder.encode_image(image)

    print("\nOriginal Image\n")
    print(image)

    print("\nSpike Tensor Shape")
    print(spike_tensor.shape)

    plot_results(
        image,
        spike_tensor,
        row=0,
        col=0
    )


if __name__ == "__main__":
    main()