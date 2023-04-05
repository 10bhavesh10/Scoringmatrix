import numpy as np
import pandas as pd
import random

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.rand(self.input_size, self.hidden_size)
        self.weights2 = np.random.rand(self.hidden_size, self.output_size)
        
    def forward(self, X):
        self.z1 = np.dot(X, self.weights1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2)
        self.y_hat = self.sigmoid(self.z2)
        return self.y_hat
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def get_weights(self):
        return np.concatenate((self.weights1.ravel(), self.weights2.ravel()))

    def set_weights(self, weights):
        w1_size = self.input_size * self.hidden_size
        self.weights1 = weights[:w1_size].reshape(self.input_size, self.hidden_size)
        self.weights2 = weights[w1_size:].reshape(self.hidden_size, self.output_size)


def fitness_func(nn, X, y):
    y_hat = nn.forward(X)
    loss = np.sum((y - y_hat) ** 2)
    return 1 / (1 + loss)

def init_pop(size, nn):
    pop = []
    for i in range(size):
        weights = nn.get_weights()
        weights += np.random.uniform(-0.5, 0.5, size=weights.shape)
        pop.append(weights)
    return pop

def cultural_algorithm(size, nn, num_iter, k, q, X, y):
    pop = init_pop(size, nn)
    best_fitness = float('-inf')
    best_solution = None
    for i in range(num_iter):
        fitness = [fitness_func(nn.set_weights(weights), X, y) for weights in pop]
        index = fitness.index(max(fitness))
        if fitness[index] > best_fitness:
            best_fitness = fitness[index]
            best_solution = nn.set_weights(pop[index])
        sorted_pop = [x for _, x in sorted(zip(fitness, pop), reverse=True)]
        num_elite = int(k*size)
        elite = sorted_pop[:num_elite]
        for j in range(num_elite, size):
            rand_elite = random.choice(elite)
            pop[j] = rand_elite + np.random.uniform(-q, q, size=rand_elite.shape)
            
    return best_solution, best_fitness

df=pd.read_csv('Bank_Personal_Loan_Modelling.csv')
df['Experience']=abs(df['Experience'])
df['Annual_CCAvg']=df['CCAvg']*12
df.drop(['ID','ZIP Code','CCAvg'],axis=1,inplace=True)
X=df.drop('Personal Loan',axis=1).values
y=df['Personal Loan'].values.reshape(-1,1)

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
input_size = X.shape[1]
hidden_size = 10
output_size = 1
nn = NeuralNetwork(input_size, hidden_size, output_size)
pop_size = 100
num_iter = 100
k = 0.2
q = 0.1

best_solution, best_fitness = cultural_algorithm(pop_size, nn, num_iter, k, q, X, y)
nn.set_weights(best_solution)
y_hat = nn.forward(X)
accuracy = np.mean((y_hat.round() == y).astype(int))
print("Best solution found with fitness = {:.4f} and accuracy = {:.2f}%".format(best_fitness, accuracy * 100))
