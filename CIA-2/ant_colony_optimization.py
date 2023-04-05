# %%
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # initialize weights randomly
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def __getitem__(self, key):
        if key == 'W1':
            return self.W1
        elif key == 'W2':
            return self.W2
        elif key == 'input_size':
            return self.input_size
        elif key == 'hidden_size':
            return self.hidden_size
        elif key == 'output_size':
            return self.output_size
        else:
            raise KeyError('Invalid key: {}'.format(key))
        

           
    def __setitem__(self, key, value):
        if key == 'W1':
            self.W1 = value
        elif key == 'W2':
            self.W2 = value
        else:
            raise KeyError('Invalid key: {}'.format(key))
        
    def forward(self, X)
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        y_hat = self.sigmoid(self.z3)
        return y_hat
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def cost(self, X, y):
        y_hat = self.forward(X).flatten()
        J = 0.5 * np.sum((y - y_hat)**2)

        return J
        
    def gradients(self, X, y):
        y_hat = self.forward(X)
        delta3 = np.multiply(-(y - y_hat), self.sigmoid_derivative(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_derivative(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        return dJdW1, dJdW2
        
    def sigmoid_derivative(self, z):
        return np.multiply(self.sigmoid(z), 1 - self.sigmoid(z))

import numpy as np

class AntColonyOptimizer:
    def __init__(self, n_ants, n_best, n_iterations, decay, alpha, beta):
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
    def optimize(self, nn, X, y):
        best_weights = None
        best_cost = float('inf')
        pheromone_W1 = np.ones(nn.W1.shape) 
        pheromone_W2 = np.ones(nn.W2.shape) 
        
        for iteration in range(self.n_iterations):
            candidate_solutions = []
            candidate_costs = []
            
            for ant in range(self.n_ants):
                perturbation_W1 = np.random.normal(0.0, 1.0, nn.W1.shape)
                candidate_W1 = nn.W1 + pheromone_W1 * perturbation_W1
                
                perturbation_W2 = np.random.normal(0.0, 1.0, nn.W2.shape)
                candidate_W2 = nn.W2 + pheromone_W2 * perturbation_W2
                
                candidate_nn = NeuralNetwork(nn.input_size, nn.hidden_size, nn.output_size)
                candidate_nn.W1 = candidate_W1
                candidate_nn.W2 = candidate_W2
                
                candidate_cost = candidate_nn.cost(X, y)
                
                candidate_solutions.append((candidate_W1, candidate_W2))
                candidate_costs.append(candidate_cost)
                
            sorted_indices = np.argsort(candidate_costs)
            sorted_solutions = [candidate_solutions[i] for i in sorted_indices]
            sorted_costs = [candidate_costs[i] for i in sorted_indices]
            best_solutions = sorted_solutions[:self.n_best]
            best_costs = sorted_costs[:self.n_best]
            
            if best_costs[0] < best_cost:
                best_weights = best_solutions[0]
                best_cost = best_costs[0]
                
            pheromone_W1 *= (1 - self.decay) 
            pheromone_W2 *= (1 - self.decay) 
            
            for solution, cost in zip(best_solutions, best_costs):
                pheromone_W1 += (self.alpha * cost /     best_cost) * solution[0] 
                pheromone_W2 += (self.alpha * cost / best_cost) * solution[1] 
            
            
            mean_W1 = np.mean([s[0] for s in candidate_solutions], axis=0)
            mean_W2 = np.mean([s[1] for s in candidate_solutions], axis=0)
            mean_nn = NeuralNetwork(nn.input_size, nn.hidden_size, nn.output_size)
            mean_nn.W1 = mean_W1
            mean_nn.W2 = mean_W2
            mean_cost = mean_nn.cost(X, y)
            
            pheromone_W1 += (self.beta * mean_cost / best_cost) * mean_W1
            pheromone_W2 += (self.beta * mean_cost / best_cost) * mean_W2       
        best_nn = NeuralNetwork(nn.input_size, nn.hidden_size, nn.output_size)
        best_nn['W1' ]= best_weights[0]
        best_nn['W2'] = best_weights[1]
        
        return best_nn

# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

import pandas as pd
df=pd.read_csv(r'C:\braindedmemory\SNU\COdes\ML tqns\EX9\Bank_Personal_Loan_Modelling.csv')
df.drop(columns=['ID','ZIP Code'],inplace=True)
y=df.pop('Personal Loan')
X=df
from sklearn.preprocessing import StandardScaler
X=StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


input_size = X.shape[1]
hidden_size = 4
output_size = 1

nn = NeuralNetwork(input_size, hidden_size, output_size)

n_ants = 10
n_best = 2
n_iterations = 10
decay = 0.5
alpha = 1
beta = 1
aco = AntColonyOptimizer(n_ants, n_best, n_iterations, decay, alpha, beta)


best_weights = aco.optimize(nn, X_train, y_train)


nn.W1= best_weights['W1'][:input_size*hidden_size].reshape(input_size, hidden_size)
nn.W2 = best_weights['W2'][:input_size*hidden_size].reshape(hidden_size, output_size)



y_pred = nn.forward(X_test)


y_pred_c = np.where(y_pred > 0.5, 1, 0)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_c))



