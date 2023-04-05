import numpy as np
import pandas as pd
from random import uniform, random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class ParticleSwarmOptimizer:
    def __init__(self, num_particles=10, maxiter=1000, verbose=False):
        self.num_particles = num_particles
        self.maxiter = maxiter
        self.verbose = verbose

    def optimize(self, costFunc, num_weights):
        x0 = [uniform(-1, 1) for _ in range(num_weights)]
        bounds = [(-1, 1) for _ in range(num_weights)]
        num_dimensions = num_weights

        class Particle:
            def __init__(self, x0):
                self.position_i = []
                self.velocity_i = []
                self.pos_best_i = []
                self.err_best_i = -1
                self.err_i = -1

                for i in range(0, num_dimensions):
                    self.velocity_i.append(uniform(-1, 1))
                    self.position_i.append(x0[i])

            def evaluate(self, costFunc):
                self.err_i = costFunc(self.position_i)

                if self.err_i < self.err_best_i or self.err_best_i == -1:
                    self.pos_best_i = self.position_i.copy()
                    self.err_best_i = self.err_i

            def update_velocity(self, pos_best_g):
                w = 0.5  # constant inertia weight (how much to weigh the previous velocity)
                c1 = 1  # cognative constant
                c2 = 2  # social constant

                for i in range(0, num_dimensions):
                    r1 = random()
                    r2 = random()

                    vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
                    vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
                    self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social

            def update_position(self, bounds):
                for i in range(0, num_dimensions):
                    self.position_i[i] = self.position_i[i] + self.velocity_i[i]

                    if self.position_i[i] > bounds[i][1]:
                        self.position_i[i] = bounds[i][1]

                    if self.position_i[i] < bounds[i][0]:
                        self.position_i[i] = bounds[i][0]

        err_best_g = -1
        pos_best_g = []

        swarm = []
        for i in range(0, self.num_particles):
            swarm.append(Particle(x0))

        i = 0
        while i < self.maxiter:
            if self.verbose:
                print(f'iter: {i:>4d}, best solution: {err_best_g:10.6f}')

            for j in range(0, self.num_particles):
                swarm[j].evaluate(costFunc)

                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g = list(swarm[j].position_i)
                    err_best_g = float(swarm[j].err_i)

            for j in range(0, self.num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i += 1

        if self.verbose:
            print('\nFINAL SOLUTION:')
            print(f'BEST WEIGHTS> {pos_best_g}')
            print(f'LOSS> {err_best_g}\n')

        return pos_best_g
    
class NeuralNetworkPSO:
    def __init__(self, n_input, n_hidden, n_output, num_particles=10, maxiter=1000, verbose=False):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.num_particles = num_particles
        self.maxiter = maxiter
        self.verbose = verbose
        
        self.pso = ParticleSwarmOptimizer(num_particles=self.num_particles, maxiter=self.maxiter, verbose=self.verbose)
        
        self.W1 = None
        self.W2 = None
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        hidden = self.sigmoid(np.dot(X, self.W1))
        output = self.sigmoid(np.dot(hidden, self.W2))
        return output
    
    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        # Initialize weights
        self.W1 = np.random.randn(self.n_input, self.n_hidden)
        self.W2 = np.random.randn(self.n_hidden, self.n_output)
        
        def costFunc(weights):
            W1 = np.reshape(weights[:self.n_input*self.n_hidden], (self.n_input, self.n_hidden))
            W2 = np.reshape(weights[self.n_input*self.n_hidden:], (self.n_hidden, self.n_output))
            
            hidden = self.sigmoid(np.dot(X_train, W1))
            output = self.sigmoid(np.dot(hidden, W2))
            
            return mean_squared_error(y_train, output)
        
        # Use PSO to optimize weights
        weights = self.pso.optimize(costFunc, self.n_input*self.n_hidden + self.n_hidden*self.n_output)
        
        self.W1 = np.reshape(weights[:self.n_input*self.n_hidden], (self.n_input, self.n_hidden))
        self.W2 = np.reshape(weights[self.n_input*self.n_hidden:], (self.n_hidden, self.n_output))
        
        # Calculate validation accuracy
        y_val_pred = self.forward(X_val)
        val_acc = accuracy_score(y_val, y_val_pred > 0.5)
        return val_acc
n_input = 10
n_hidden = 5
n_output = 1
num_particles = 10
maxiter = 100   
verbose = True

nn_pso = NeuralNetworkPSO(n_input=11, n_hidden=11, n_output=1, num_particles=100, maxiter=200, verbose=True)


val_acc = nn_pso.fit(X, y)
print("ACCURACY: ",val_acc)
