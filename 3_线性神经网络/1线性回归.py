import math
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# n = 10000
# a = torch.ones([n])
# b = torch.ones([n])

# start_time = time.time()
# d = a+b
# end_time = time.time()
# print(f'{end_time-start_time:.7f} sec')

def normal(x, mu, sigma):
    p = 1/math.sqrt(2*math.pi*sigma**2)
    return p*np.exp(-0.5*(x-mu)**2/sigma**2)

x = np.arange(-7, 7, 0.01)
params = [(0,1),(0,2),(3,1)]
plt.figure(figsize=(4.5,2.5))
for mu, sigma in params:
    plt.plot(x,normal(x, mu, sigma),label=f'mu={mu},sigma={sigma}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.show()