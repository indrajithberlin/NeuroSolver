import matplotlib.pyplot as plt


beta = 0.9
threshold = 1.0
membrane = 0.0

time_steps = 50


input_current = [0.2] * time_steps


membrane_history = []
spike_history = []


for time, current in enumerate(input_current):

    membrane = beta * membrane + current

    membrane_history.append(membrane)

    if membrane >= threshold:

        spike = 1

        print(
            f"Time {time:2d} | "
            f"Membrane: {membrane:.4f} | "
            f"SPIKE!"
        )

        # Reset after firing
        membrane = 0.0

    else:

        spike = 0

        print(
            f"Time {time:2d} | "
            f"Membrane: {membrane:.4f} | "
            f"Spike: 0"
        )

    spike_history.append(spike)


plt.figure(figsize=(10, 4))

plt.plot(
    range(time_steps),
    input_current,
    label="Input Current"
)

plt.xlabel("Time Step")
plt.ylabel("Input Current")

plt.title("LIF Neuron: Input Current Over Time")

plt.legend()
plt.grid(True)

plt.show()


plt.figure(figsize=(10, 4))

plt.plot(
    range(time_steps),
    membrane_history,
    label="Membrane Potential"
)

plt.axhline(
    y=threshold,
    linestyle="--",
    label="Firing Threshold"
)

plt.xlabel("Time Step")
plt.ylabel("Membrane Potential")

plt.title("LIF Neuron: Membrane Potential Over Time")

plt.legend()
plt.grid(True)

plt.show()


plt.figure(figsize=(10, 4))

plt.stem(
    range(time_steps),
    spike_history
)

plt.xlabel("Time Step")
plt.ylabel("Spike")

plt.yticks([0, 1])

plt.title("LIF Neuron: Output Spike Train")

plt.grid(True)

plt.show()