import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv('Bank_Personal_Loan_Modelling.csv')
df['Experience']=abs(df['Experience'])
df['Annual_CCAvg']=df['CCAvg']*12
df.drop(['ID','ZIP Code','CCAvg'],axis=1,inplace=True)
X=df.drop('Personal Loan',axis=1).values
y=df['Personal Loan'].values.reshape(-1,1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


input_size = X_train.shape[1]
hidden_size = 5
output_size = 1

population_size = 30
mutation_rate = 0.1
generations = 500

def fitness(individual):
    W1 = np.array(individual[:input_size*hidden_size]).reshape((input_size, hidden_size))
    b1 = np.array(individual[input_size*hidden_size:input_size*hidden_size+hidden_size])
    W2 = np.array(individual[input_size*hidden_size+hidden_size:]).reshape((hidden_size, output_size))
    b2 = np.array([0.0])
    Z1 = np.dot(X_train, W1) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(A1, W2) + b2
    y_pred = np.round(1 / (1 + np.exp(-Z2)))
    accuracy = np.sum(y_pred == y_train) / len(y_train)
    return accuracy

population = []
for i in range(population_size):
    individual = np.concatenate([np.random.randn(input_size*hidden_size), np.random.randn(hidden_size), np.random.randn(hidden_size*output_size)])
    population.append(individual)
for generation in range(generations):
    fitness_scores = [fitness(individual) for individual in population]
    parent1 = population[fitness_scores.index(max(fitness_scores))]
    parent2 = population[fitness_scores.index(sorted(fitness_scores)[-2])]
    offspring = []
    for i in range(population_size):
        child = []
        for j in range(len(parent1)):
            if random.random() < 0.5:
                child.append(parent1[j])
            else:
                child.append(parent2[j])
        for j in range(len(child)):
            if random.random() < mutation_rate:
                child[j] += np.random.randn()
        offspring.append(np.array(child))
    population = offspring
    print("Generation:", generation+1)
    print("Best fitness score:", max(fitness_scores))
best_individual = population[fitness_scores.index(max(fitness_scores))]
W1 = np.array(best_individual[:input_size*hidden_size]).reshape((input_size, hidden_size))
b1 = np.array(best_individual[input_size*hidden_size:input_size*hidden_size+hidden_size])
W2 = np.array(best_individual[input_size*hidden_size+hidden_size:]).reshape((hidden_size, output_size))
b2 = np.array([0.0])
Z1 = np.dot(X_test, W1) + b1
A1 =np.tanh(Z1)
Z2 = np.dot(A1, W2) + b2
y_pred = np.round(1 / (1 + np.exp(-Z2)))
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Test accuracy:", accuracy)