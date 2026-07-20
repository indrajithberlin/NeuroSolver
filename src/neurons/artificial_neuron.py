import math
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + math.exp(-x))



weight = 2.0
bias = -1.0

inputs = []
outputs = []



for i in range(101):

    x = i / 100

    z = (x * weight) + bias

    output = sigmoid(z)

    inputs.append(x)
    outputs.append(output)


print("Artificial Neuron\n")

for i in range(0, 101, 20):

    print(
        f"Input: {inputs[i]:.2f} | "
        f"Output: {outputs[i]:.4f}"
    )


plt.figure(figsize=(10, 5))

plt.plot(
    inputs,
    outputs,
    label="Neuron Output"
)

plt.xlabel("Input Value")
plt.ylabel("Output Value")

plt.title(
    "Artificial Neuron: Input vs Output"
)

plt.legend()
plt.grid(True)

plt.show()