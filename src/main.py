import numpy as np
import pandas as pd
import warnings
import santa_chromo



warnings.filterwarnings("ignore")

POPULATION_SIZE = 2
OFFSPRING_SIZE = 1
MUTATION_RATE = 1.0
EPOCHS = 10

print("initialization")
santa = santa_chromo.SantaChromo(POPULATION_SIZE)

record = []
for i in range(EPOCHS):
    new_population = []
  
    for j in range(OFFSPRING_SIZE):
        new_population.extend(santa.crossover_random())
    
    new_population.extend(santa.mutate_all(MUTATION_RATE))

    santa.population.extend(new_population)

    santa.eliminate(POPULATION_SIZE)

    santa.ranking()
    best = santa.population[0][1]
    worst  = santa.population[-1][1]
    average = 0
    for p in santa.population:
        average += p[1]
    average /= len(santa.population)
    record.append([best, average, worst])

    if i % 10 == 0:
        data = pd.DataFrame(record)
        data.to_csv("output/learning_curve.csv", index = False)

    print(i, best, average, worst)


santa.save("output/result.csv")