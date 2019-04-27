import numpy as np
import matplotlib

matplotlib.use('TKAgg')

from matplotlib import pyplot as plt

#%% plot something
x = np.arange(50)
y = x**2

# plot
fig, ax = plt.subplots(figsize = (5, 5))
ax.plot(x, y)
plt.show()

