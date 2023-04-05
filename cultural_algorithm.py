import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
from sklearn.preprocessing import StandardScaler


POP_SIZE = 50
NUM_ITERATIONS = 100
BELIEF_SPACE_SIZE = 5
MUTATION_RATE = 0.01
MUTATION_STD = 0.1
NUM_INPUTS = 2
NUM_CLASSES = 2
HIDDEN_LAYER_SIZE =1

df=pd.read_csv('Bank_Personal_Loan_Modelling.csv')
df['Experience']=abs(df['Experience'])
df['Annual_CCAvg']=df['CCAvg']*12
df.drop(['ID','ZIP Code','CCAvg'],axis=1,inplace=True)
X=df.drop('Personal Loan',axis=1).values
y=df['Personal Loan'].values.reshape(-1,1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def create_model(weights):

    model = Sequential()
    model.add(Dense(HIDDEN_LAYER_SIZE, input_dim=NUM_INPUTS))
    model.add(Activation('sigmoid'))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    model.set_weights(weights)
    return model

def softmax(x):
 
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

def evaluate_fitness(individual, X, y):
    model = create_model(individual)
    y_pred = model.predict(X)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y, axis=1)
    accuracy = np.mean(y_pred_labels == y_true_labels)
    return accuracy


population = [np.random.uniform(low=-1, high=1, size=(NUM_INPUTS*HIDDEN_LAYER_SIZE+HIDDEN_LAYER_SIZE*NUM_CLASSES,))
              for _ in range(POP_SIZE)]

belief_space = [population[np.argmax([evaluate_fitness(individual, X_train, y_train) for individual in population])]]

for i in range(NUM_ITERATIONS):
    fitness_scores = [evaluate_fitness(individual, X_train, y_train) for individual in population]
    population = [individual for _, individual in sorted(zip(fitness_scores, population), reverse=True)]
    if i % BELIEF_SPACE_SIZE == 0:
        best_individual = population[0]
        if evaluate_fitness(best_individual, X_train, y_train) > evaluate_fitness(belief_space[-1], X_train, y_train):
            belief_space.append(best_individual)
            belief_space = belief_space[-BELIEF_SPACE_SIZE:]
    new_population = []
    for j in range(POP_SIZE):
        belief = belief_space[np.random.randint(len(belief_space))]
        mutation = np.random.normal(loc=0, scale=MUTATION_STD)
    mutated_weights = belief + MUTATION_RATE * mutation
    new_population.append(mutated_weights)
population = new_population
best_individual = belief_space[-1]
model = create_model(best_individual)
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
test_accuracy = np.mean(y_pred_labels == y_true_labels)
print('Test accuracy:', test_accuracy)