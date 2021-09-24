import random
import numpy as np
from deap import creator, base, tools, algorithms
import math
from scipy.spatial import distance_matrix
import pandas as pd

numCities = 127
cityArr = []

def popArr(fileName):
    with open(fileName, "r") as f:
        for line in f:
            line = line.strip().split()
            cityArr.append([int(line[0]),int(line[1])])

def makeDM():
    global dm

    df = pd.DataFrame(cityArr, columns=['x', 'y'])
    dm = pd.DataFrame(distance_matrix(df.values, df.values),
                      index=df.index, columns=df.index)
    
def calcDist(indiv):
    dist = 0
    distance = dm[indiv[-1]][indiv[0]]
    for city1, city2 in zip(indiv[0:-1], indiv[1:]):
        dist += dm[city1][city2]
         
    return dist,
    
def calcLength(indiv):
    length = 0
    
    for i in range(1,numCities):
        length += math.sqrt((cityArr[indiv[i-1]][0]
                             - cityArr[indiv[i]][0])**2)
        length += math.sqrt((cityArr[indiv[i-1]][1]
                             - cityArr[indiv[i]][1])**2)

    length += math.sqrt((cityArr[numCities-1][0]
                         - cityArr[0][0])**2)
    length += math.sqrt((cityArr[numCities-1][1]
                         - cityArr[0][1])**2)
    return length,

popArr("cityData.txt")
makeDM()

creator.create("fitnessMin", base.Fitness, weights = (-1.0,))
creator.create("Individual", np.ndarray, fitness = creator.fitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample,range(numCities),numCities)
toolbox.register("individual", tools.initIterate,
                 creator.Individual, toolbox.indices)

toolbox.register("population", tools.initRepeat, list,toolbox.individual)

makeDM()

toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes)
toolbox.register("select", tools.selTournament, tournsize=15)
toolbox.register("evaluate", calcDist)

def main():
    pop = toolbox.population(n=600)
    hof = tools.HallOfFame(1, similar=np.array_equal)
    CXPB = 0.9
    numEvals = 0
    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        numEvals += 1
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))
     
    fits = [ind.fitness.values[0] for ind in pop]
    # Variable keeping track of the number of generations
    g = 0


    # Begin the evolution
    while min(fits) > 118282 and numEvals < 1000000:
        # A new generation
        g += 1
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                #toolbox.mate(child1, child2,0.5)
                toolbox.mate(child1, child2)
            
                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values
       
        for mutant in offspring:
            # mutate an individual with probability MUTPB
            toolbox.mutate(mutant, 1/numCities)
            del mutant.fitness.values
       

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            numEvals += 1
            
        print("  Evaluated %i individuals" % len(invalid_ind))
        print("  Evaluated %i total individuals" % numEvals)
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        hof.update(pop)

        #stat calculation
        length = len(pop) #sets length var for use in stat calc
        mean = sum(fits) / length #calc mean
        #stddev calculation
        #sum2 = sum(x*x for x in fits) 
        #std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min Fitness: %s" % min(fits))
        #print("  Max %s" % max(fits))
        print("  Avg Fitness: %s" % mean)
        #print("  Std %s" % std)
       
        
    print (hof[0])
    
main()


