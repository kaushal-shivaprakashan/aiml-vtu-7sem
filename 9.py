import numpy as np
import math
import matplotlib.pyplot as plt
x = np.linspace(0, 2 * math.pi, 100)
y = np.sin(x) + 0.3 * np.random.randn(100)
plt.plot(x, y)

import statsmodels.api as sm
lowess = sm.nonparametric.lowess(y, x, frac=.3)
lowess_x = list(zip(*lowess))[0]
lowess_y = list(zip(*lowess))[1]
plt.plot(lowess_x, lowess_y)
plt.show()



################################################

