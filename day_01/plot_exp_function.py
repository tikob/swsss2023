import matplotlib.pyplot as plt  # here is the good stuff
import numpy as np

x = np.linspace(0, 1)  # default num is 50
plt.plot(x, np.exp(x))
plt.xlabel('$0 \leq x < 1$')
plt.ylabel(r'$e^x$')
plt.title('Exponential function')
plt.show()
