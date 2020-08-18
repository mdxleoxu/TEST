import numpy as np
import pandas as pd
import warnings
import santa_chromo



warnings.filterwarnings("ignore")

POPULATION_SIZE = 20
OFFSPRING_SIZE = 15
MUTATION_RATE = 1
EPOCHS = 1000000

parameters = input("parameters summary: ")

print("initialization")
santa = santa_chromo.SantaChromo(POPULATION_SIZE)
santa.optimize_all(10)
record = []

for i in range(EPOCHS):
    new_population = []
    new_population.extend(santa.crossover_by_fittness(OFFSPRING_SIZE))
    new_population.extend(santa.mutate_all(MUTATION_RATE))
    santa.population.extend(new_population)
    santa.eliminate(POPULATION_SIZE)
    santa.ranking()
    status = santa.get_status()
    record.append(status)
    
    if i % 10 == 0:
        data = pd.DataFrame(record)
        data.to_csv("output/" + parameters + "_learning_curve.csv", index = False)
    print(i,status)
    

santa.save("output/2p_result.csv")