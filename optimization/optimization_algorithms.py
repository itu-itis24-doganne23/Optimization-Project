import numpy as np
import matplotlib.pyplot as plt
import csv

#GENETIC ALGORITHM
#Select Methods
def truncation_selection(data,k):
    p = np.argsort(data)
    res = [p[np.random.choice(k,2)] for _ in data]
    return res
def tournament_selection(data,k):
    def get_parent():
        candidates = np.random.permutation(len(data))[:k]
        best = candidates[np.argmin(data[candidates])]
        return best
    return [[get_parent(),get_parent()] for _ in data]
def roulette_wheel_selection(data):
    data = np.max(data) - data
    data = data / np.sum(data)
    return [np.random.choice(len(data),2,p=data) for _ in data]

#Crossover Methods
def single_point_crossover(a,b):
    i = np.random.choice(len(a),1)[0]
    a = a[:i]
    b = b[i:]
    return np.concatenate((a,b))
def two_point_crossover(a,b):
    i,j = np.random.choice(len(a),2)
    if i > j:
        i,j = j,i
    p1 = a[:i]
    p2 = b[i:j]
    p3 = a[j:]
    return np.concatenate((p1,p2,p3))
def uniform_crossover(a,b,p):
    return [int(u) if np.random.rand() > p else int(v) for (u,v) in zip(a,b)]
def interpolation_crossover(a,b,lambda_):
    return (1-lambda_)*a + lambda_*b
    
#Mutation Methods
def distribution_mutation(child,lambda_,D):
    return [v + np.random.choice(D) if np.random.rand() < lambda_ else v for v in child]
def gaussian_mutation(child,mu):
    return distribution_mutation(child,1,np.random.normal(0,mu,1000))

def rand_population_uniform(m, a, b):
    d = len(a)
    return [a + np.random.rand(d)*(b - a) for _ in range(m)]

def genetic_algorithm(f,m,a,b,max_iter,seed):
    np.random.seed(seed)  
    population = rand_population_uniform(m, a, b)


    for _ in range(max_iter):
        fitness = np.array([f(ind[0],ind[1]) for ind in population])
        parents_indices = truncation_selection(fitness, 10)
        new_population = []
        for idx1, idx2 in parents_indices:
            parent1 = population[idx1]
            parent2 = population[idx2]
            child = single_point_crossover(parent1, parent2)
            child = gaussian_mutation(child, 0.5)
            new_population.append(np.array(child))

        population = new_population

    fitness = np.array([f(ind[0],ind[1]) for ind in population])
    best_idx = np.argmin(fitness)
    best_individual = population[best_idx]
    return best_individual

#PARTICLE SWARM OPTIMIZATION
class Particle:
    def __init__(self,x,v):
        self.x = x
        self.v = v
        self.x_best = self.x.copy()
    
def particle_swarm_optimization(f,population,seed,k_max,w=0.7,c1=1.5,c2=1.5):
    np.random.seed(seed)
    n = len(population[0].x)
    x_best = population[0].x_best.copy()
    y_best = float("inf")
    for p in population:
        y = f(p.x[0],p.x[1])
        if y < y_best:
            x_best,y_best = p.x.copy(),y
    
    for k in range(k_max):
        for p in population:
            r1,r2 = np.random.rand(n),np.random.rand(n)
            p.v = w*p.v + c1*r1*(p.x_best-p.x) + c2*r2*(x_best-p.x)
            p.v = np.clip(p.v, -3, 3)
            p.x += p.v
            p.x = np.clip(p.x, -10, 10)
            y = f(p.x[0],p.x[1])

            if y < y_best:
                x_best,y_best = p.x.copy(),y
            if y < f(p.x_best[0],p.x_best[1]):
                p.x_best = p.x.copy()
    fitness = np.array([f(ind.x[0],ind.x[1]) for ind in population])
    best_idx = np.argmin(fitness)
    best_individual = population[best_idx]
    return best_individual.x

def create_population(m,lower_bound,upper_bound,seed):
    np.random.seed(seed)
    population = []
    for _ in range(m):
        x = np.random.uniform(lower_bound,upper_bound,2)
        v = np.random.uniform(-1,1,2)
        particle = Particle(x,v)
        population.append(particle)
    return population
