# python tanh.py  
import numpy as np
import matplotlib.pyplot as plt

plt.plot(np.arange(-5, 5, 0.2), np.tanh(np.arange(-5,5,0.2)))
plt.grid(True)
plt.show()