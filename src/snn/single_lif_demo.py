import torch
import torch.nn as nn
import snntorch as snn
import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------

num_inputs = 5
num_outputs = 3
time_steps = 40

# -------------------------
# Layers
# -------------------------

fc = nn.Linear(num_inputs, num_outputs)

lif = snn.Leaky(beta=0.9)

# -------------------------
# Initial membrane potential
# -------------------------

mem = lif.init_leaky()

# -------------------------
# Store outputs
# -------------------------

spike_record = []
mem_record = []

# -------------------------
# Random spike input
# -------------------------

torch.manual_seed(42)

for step in range(time_steps):

    x = torch.randint(0, 2, (1, num_inputs)).float()

    current = fc(x)

    spike, mem = lif(current, mem)

    spike_record.append(spike.detach().numpy())

    mem_record.append(mem.detach().numpy())

# -------------------------
# Convert to arrays
# -------------------------

import numpy as np

spike_record = np.array(spike_record)
mem_record = np.array(mem_record)

# -------------------------
# Plot membrane potential
# -------------------------

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(mem_record[:,0,0])

plt.title("Neuron 1 Membrane Potential")

plt.xlabel("Time")

plt.ylabel("Voltage")

# -------------------------
# Plot spikes
# -------------------------

plt.subplot(1,2,2)

plt.step(
    range(time_steps),
    spike_record[:,0,0],
    where="mid"
)

plt.ylim(-0.2,1.2)

plt.title("Neuron 1 Spike Output")

plt.xlabel("Time")

plt.ylabel("Spike")

plt.tight_layout()

plt.show()