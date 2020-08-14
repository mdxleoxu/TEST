import santa_chromo
import pandas as pd

santa = santa_chromo.SantaChromo(1)
santa.optimize_all(5000)
pd.DataFrame(santa.population[0][0],columns=["choice"]).to_csv("output\local_min.csv")