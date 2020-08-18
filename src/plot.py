import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""
record = pd.read_csv("output/_learning_curve.csv").to_numpy()
record = np.transpose(record)
plt.plot(record[0],label="")
"""

record = pd.read_csv("output/t_learning_curve.csv").to_numpy()
record = np.transpose(record)
plt.plot(record[0],label="t")

record = pd.read_csv("output/tournament_learning_curve.csv").to_numpy()
record = np.transpose(record)
plt.plot(record[0],label="tournament")


record = pd.read_csv("output/bigger_init_learning_curve.csv").to_numpy()
record = np.transpose(record)
plt.plot(record[0],label="bigger initial set")

plt.xlabel("Generation")
plt.ylabel("cost of the best")
plt.legend(loc = 'upper right')
plt.savefig("output/learning_curve.jpg", dpi=1000)
plt.show()
