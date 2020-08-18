import numpy as np
import pandas as pd
import warnings
import santa_chromo



warnings.filterwarnings("ignore")

POPULATION_SIZE = 100
OFFSPRING_SIZE = 100
MUTATION_RATE = 0.2
MUTATION_SIZE = 5
EPOCHS = 1000

parameters = input("parameters summary: ")

print("initialization")
santa = santa_chromo.SantaChromo(POPULATION_SIZE)
santa.optimize_all(10)
record = []

for i in range(EPOCHS):
    status = santa.get_status()
    record.append(status)
    new_population = []
    new_population.extend(santa.crossover_by_fitness(OFFSPRING_SIZE))
    new_population.extend(santa.mutate_all(MUTATION_RATE, MUTATION_SIZE))
    santa.population.extend(new_population)
    santa.eliminate(POPULATION_SIZE)
    santa.ranking()
    if i % 10 == 0:
        data = pd.DataFrame(record)
        data.to_csv("output/" + parameters + "_learning_curve.csv", index = False)
    if i % 100 == 0:
        print(parameters)
        santa.save("output/" + parameters+"_sol.csv")
    
    print(i,status)
    

