import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

record = pd.read_csv("output/1point_hill_learning_curve.csv").to_numpy()
record = np.transpose(record)
plt.plot(record[0],label="1-point crossover")

record = pd.read_csv("output/2point_hill_learning_curve.csv").to_numpy()
record = np.transpose(record)
plt.plot(record[0],label="2-point crossover")

record = pd.read_csv("output/uniform_hill_learning_curve.csv").to_numpy()
record = np.transpose(record)
plt.plot(record[0],label="unifrom crossover")

record = pd.read_csv("output/uni_random_init_learning_curve.csv").to_numpy()
record = np.transpose(record)
plt.plot(record[0],label="unifrom crossover with more random initial solution")

record = pd.read_csv("output/uniform_hill_step_100_learning_curve.csv").to_numpy()
record = np.transpose(record)
plt.plot(record[0],label="unifrom crossover w/ hill-climbing 100step")

record = pd.read_csv("output/uni_init_hill_learning_curve.csv").to_numpy()
record = np.transpose(record)
#plt.plot(record[0],label="unifrom crossover w/ hill-climbing_initialize")

record = pd.read_csv("output/uni_hybrid_hill_learning_curve.csv").to_numpy()
record = np.transpose(record)
plt.plot(record[0],label="unifrom crossover with hill-climbing for every solution")

plt.xlabel("Generation")
plt.ylabel("cost of the best")
plt.legend(loc = 'upper right')
plt.savefig("output/learning_curve.jpg", dpi=1000)
plt.show()
