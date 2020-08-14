import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

record = pd.read_csv("output/learning_curve.csv").to_numpy()
record = np.transpose(record)
plt.plot(record[0],label="without hill-climbing best")

record = pd.read_csv("output/hill_learning_curve.csv").to_numpy()
record = np.transpose(record)
#plt.plot(record[1],label="without hill-climbing average")

plt.xlabel("Generation")
plt.ylabel("cost")
plt.legend(loc = 'upper right')
plt.show()