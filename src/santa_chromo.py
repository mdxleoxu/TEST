# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
"""
SciPy license
Copyright © 2001, 2002 Enthought, Inc.
All rights reserved.

Copyright © 2003-2019 SciPy Developers.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name of Enthought nor the names of the SciPy Developers may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import copy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import multiprocessing as mp
from scipy.optimize import linear_sum_assignment
from numba import jit


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
class SantaChromo:

    population = []
    family_size = 0
    family_chioce = []
    family_member = []
    cost_matrix = [[]]
    accountiog_matrix = [[]]
    cost = []
    def __init__(self, population_size):

        fpath = '.\\data\\family_data.csv'
        data = pd.read_csv(fpath, index_col='family_id')
        self.family_size = data.shape[0]
        self.family_chioce    = data[["choice_0", "choice_1", "choice_2", "choice_3", "choice_4", "choice_5", "choice_6", "choice_7", "choice_8", "choice_9", ]].to_numpy()
        self.family_member    = data["n_people"].to_numpy()

        #family_cost_matrix: the cost on different choices with respect to the size of the family
        self.family_cost_matrix = [
            [0,
            50,
            50 + 9*i,
            100+ 9*i,
            200+ 9*i,
            200+18*i, 
            300+18*i, 
            300+36*i, 
            400+36*i, 
            500+36*i+199*i, 
            500+36*i+398*i
            ] for i in range(max(self.family_member)+1)
        ]

        # cost_matrix: the cost of different day with respect to family 1~ family5000
        self.cost_matrix = [[0 for i in range(100)] for i in range(5000)]
        for i in range(5000):
            for j in range(100):
                self.cost_matrix[i][j] = self.family_cost_matrix[self.family_member[i]][10]
        for i in range(5000):
            for j in range(10):
                self.cost_matrix[i][self.family_chioce[i][j]-1] = self.family_cost_matrix[self.family_member[i]][j]
        
        """
        uniform_cost_matrix = [[0 for i in range(100)] for i in range(5000)]
        for i in range(5000):
            for j in range(100):
                uniform_cost_matrix[i][j] = self.family_cost_matrix[self.family_member[i]][10]
                uniform_cost_matrix[i][j] = self.family_cost_matrix[1][10]
        for i in range(5000):
            for j in range(10):
                uniform_cost_matrix[i][self.family_chioce[i][j]-1] = self.family_cost_matrix[self.family_member[i]][j]
                uniform_cost_matrix[i][self.family_chioce[i][j]-1] = self.family_cost_matrix[9][j]
        """
        
        # accounting cost
        self.accounting_matrix = np.zeros((500, 500))
        for i in range(500):
            for j in range(500):
                self.accounting_matrix[i][j] = (i-125) / 400 * np.power(i, 0.5+abs(i-j)/50)
        
        """
        linear_sum_assignment: Hungarian method
        assign 100 day to 100 different family at a time
        until all family are assigned 
        """
        self.population = []
        for i in range(population_size):    
            seq = np.random.permutation(5000)
            tmp_cost_matrix = [copy.deepcopy(self.cost_matrix[j]) for j in seq]
            tmp_cost_matrix = np.transpose(tmp_cost_matrix)
            days_visitors  = [0 for j in range(100)]
            chromo = [-1 for j in range(5000)]
            while -1 in chromo:
                row_ind, col_ind=linear_sum_assignment(tmp_cost_matrix)
                for j in range(len(col_ind)):
                    if days_visitors[j] <= 300:
                        ori_seq = seq[col_ind[j]]
                        chromo[ori_seq] = j
                        tmp_cost_matrix[:, col_ind[j]] += 100000000000
                        days_visitors[j] += self.family_member[ori_seq]
            self.population.append((chromo, self.cost_function(chromo)))
        
        return

    @jit(looplift=False, fastmath=True)
    def cost_function(self, chromosome):
        days_visitors  = np.zeros(100, dtype=int)
        cost = 0
        for i in range(self.family_size): 
            day = chromosome[i]
            days_visitors[day] += self.family_member[i]
            cost += self.cost_matrix[i][day]
        last_day = days_visitors[0]
        for day in days_visitors:
            cost += self.accounting_matrix[day][last_day]
        for visitors in days_visitors:
            if visitors > 300 or visitors < 125:
                cost += 100000000000
        
        return cost

    def crossover(self, p1, p2):
        """
        choose a random position x
        child 1 = parent 1[0:x] + parent 2[x:0]
        child 2 = parent 2[0:x] + parent 1[x:0]
        """
        x = np.random.randint(self.family_size-1)+1
        c1 = copy.deepcopy(p1[0][:x])
        c1.extend(copy.deepcopy(p2[0][x:]))
        c2 = copy.deepcopy(p2[0][:x])
        c2.extend(copy.deepcopy(p1[0][x:]))
        return (c1,self.cost_function(c1)), (c2,self.cost_function(c2))

    def crossover_random(self):
        """
        choose 2 random parent p1, p2
        choose a random position z
        child 1 = parent 1[0:z] + parent 2[z:0]
        child 2 = parent 2[0:z] + parent 1[z:0]
        """
        x = np.random.randint(len(self.population))
        y = np.random.randint(len(self.population)) 
        p1 = self.population[x]
        p2 = self.population[y]
        z = np.random.randint(self.family_size-1)+1
        c1 = copy.deepcopy(p1[0][:z])
        c1.extend(copy.deepcopy(p2[0][z:]))
        c2 = copy.deepcopy(p2[0][:z])
        c2.extend(copy.deepcopy(p1[0][z:]))
        return [(c1,self.cost_function(c1)), (c2,self.cost_function(c2))]

    def crossover_2point_random(self):
        """
        choose 2 random parent p1, p2
        choose 2 random position z1, z2
        0<z1<z2<5000
        child 1 = parent 1[0:z1] + parent 2[z1:z2] + parent 1[z2:0]
        child 2 = parent 2[0:z2] + parent 1[z1:z2] + parent 2[z2:0]
        """
        x = np.random.randint(len(self.population))
        y = np.random.randint(len(self.population)) 
        p1 = self.population[x][0]
        p2 = self.population[y][0]
        z = np.random.choice(self.family_size-1, 2)
        z[0] += 1
        z[1] += 1
        z.sort()
        c1 = copy.deepcopy(p1[:z[0]])
        c1.extend(copy.deepcopy(p2[z[0]:z[1]]))
        c1.extend(copy.deepcopy(p1[z[1]:]))
        c2 = copy.deepcopy(p2[:z[0]])
        c2.extend(copy.deepcopy(p1[z[0]:z[1]]))
        c2.extend(copy.deepcopy(p2[z[1]:]))
        return [(c1,self.cost_function(c1)), (c2,self.cost_function(c2))]

    def crossover_2point_fitting(self):
        """
        choose 2 random parent p1, p2
        choose 2 random position z1, z2
        0<z1<z2<5000
        child 1 = parent 1[0:z1] + parent 2[z1:z2] + parent 1[z2:0]
        child 2 = parent 2[0:z2] + parent 1[z1:z2] + parent 2[z2:0]
        """
        x = np.random.randint(len(self.population))
        y = np.random.randint(len(self.population)) 
        p1 = self.population[x][0]
        p2 = self.population[y][0]
        z = np.random.choice(self.family_size-1, 2)
        z[0] += 1
        z[1] += 1
        z.sort()
        
        c1 = copy.deepcopy(p1[:z[0]])
        c1.extend(copy.deepcopy(p2[z[0]:z[1]]))
        c1.extend(copy.deepcopy(p1[z[1]:]))
        c2 = copy.deepcopy(p2[:z[0]])
        c2.extend(copy.deepcopy(p1[z[0]:z[1]]))
        c2.extend(copy.deepcopy(p2[z[1]:]))
        
        return [(c1,self.cost_function(c1)), (c2,self.cost_function(c2))]


    def crossover_uniform_random(self):
        """
        choose 2 random parent p1, p2
        every "bit" is chooced from p1 or p2 independently
        """
        x = np.random.randint(len(self.population))
        y = np.random.randint(len(self.population)) 
        p = [self.population[x][0], self.population[y][0]]
        c1 = []
        c2 = []
        for i in range(self.family_size):
            z = np.random.permutation(2)
            c1.append(p[z[0]][i])
            c2.append(p[z[1]][i])
        return [(c1,self.cost_function(c1)), (c2,self.cost_function(c2))]

    def mutate(self, sol):
        # randomly assign another day to a random family
        new_chromo = copy.deepcopy(sol[0])
        x = np.random.randint(self.family_size)
        y = np.random.randint(100)
        new_chromo[x] = y
        return (new_chromo, self.cost_function(new_chromo))
    
    def mutate_by_Hungarian(self, sol):
        new_chromo = copy.deepcopy(sol[0])
        for i in range(10):
            seq = np.random.choice(range(5000),100)
            tmp_cost_matrix = [copy.deepcopy(self.cost_matrix[j]) for j in seq]
            tmp_cost_matrix = np.transpose(tmp_cost_matrix)        
            row_ind,col_ind=linear_sum_assignment(tmp_cost_matrix)
            for j in range(len(col_ind)):    
                ori_seq = seq[col_ind[j]]
                new_chromo[ori_seq] = j
        return (new_chromo, self.cost_function(new_chromo))

    def foo(self):
        for i in range(len(self.population)):
            tmp = self.mutate_by_Hungarian(self.population[i])
            if tmp[1] < self.population[i][1]:
                print(tmp[1])
                self.population[i] = tmp
            return

    def mutate_all(self, mutate_rate):
        # mutate every solution in population with probability of (mutate_rate)
        new_population = []
        for sol in self.population:
            if np.random.random() < mutate_rate:
                new_population.append(self.mutate(sol))
        return new_population
    
    def hill_climbing(self, sol, candidates):
        # search and replace the solution utill (candidates) consecutiuve solutions are not better
        best_chromo = copy.deepcopy(sol[0])
        best = sol[1]
        unchanged = 0
        while(unchanged < candidates):
            unchanged += 1
            new_chromo = copy.deepcopy(best_chromo)
            x = np.random.randint(self.family_size)
            y = np.random.randint(100)
            new_chromo[x] = y
            performance = self.cost_function(new_chromo)
            if best > performance:
                best = performance
                best_chromo = new_chromo
                unchanged = 0
        return (best_chromo,best)

    def optimize_all(self, candidates):
        # search and replace all the solution
        for i in range(len(self.population)):
            self.population[i] = self.hill_climbing(self.population[i],candidates)
        return 

    def optimize_all_parallel(self, candidates):
        """
        the parallel computing version of optimize_all
        , due to @jit decorator of cost_funtion
        , it's now non-parallel now...
        """  
        pool = mp.Pool(mp.cpu_count())
        for i in range(len(self.population)):
            self.population[i] = pool.apply_async(self.hill_climbing,args=(self.population[i],candidates)).get()
        pool.close()
        pool.join()
        return 

    def ranking(self):
        # sort the population by ranking
        self.population.sort(key = lambda x: x[1])
        return

    def eliminate(self, population_size):
        # keep top (population_size) people and dump others
        self.ranking()
        self.population = self.population[:population_size]
        return

    def show_best_worst(self):
        # print the best and worst 5 score
        self.ranking()
        if len(self.population)<5:
            print("all:")
            for p in self.population:
                print(p[1])
        else:
            print("best 5:")
            for p in self.population[:5]:
                print(p[1])
            print("worst 5:")
            for p in self.population[-5:]:
                print(p[1]) 
        return       

    def save(self, path):
        data = pd.DataFrame([x[0] for  x in self.population])
        data.to_csv(path, index = False)

    def show_all(self):
        self.ranking()
        print("all:")
        for p in self.population:
            print(p[1])

    def load(self, path):
        data = pd.read_csv(path)
        data = data.values.tolist()
        self.population = [(x, self.cost_function(x)) for x in data]
        return