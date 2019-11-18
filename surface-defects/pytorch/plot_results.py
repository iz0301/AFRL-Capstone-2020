import torch
import numpy as np
import matplotlib.pyplot as plt

n = 10 # Averaging window
losses = torch.load("losses.np")
losses = np.convolve(losses, np.ones(n))
losses = losses[(n-1):-(n-1)]
plt.figure()
plt.plot(range(len(losses)), losses)
plt.show()
