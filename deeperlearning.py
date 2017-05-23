#Using Grid Search to Find Optimal Parameters for Deep Learning Models
#Loading important libraries
from keras.models import Sequential
from keras.layers import Dense
import numpy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

#Creating the function required for the keras classifier
def create_model(optimizer='rmsprop', init='glorot_uniform'):
    #Create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    #Compiling the model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
	
#Fixing reproducibilty
seed = 7
numpy.random.seed(seed)

#Loading the dataset
dataset = numpy.loadtxt('pima-indians-diabetes.data.csv', delimiter=',')

#Spliting the inputs and outputs
X = dataset[:,0:8]
Y = dataset[:,8]

#Creating the model with KerasClassifier
model = KerasClassifier(build_fn=create_model, verbose=0)

#Grid search epochs, batch size and optimizer
optimizers = ['rmsprop', 'adam']
inits = ['glorot_uniform', 'normal', 'uniform']
epochs = [50, 100, 150, 200]
batched = [5, 10, 20, 30]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batched, init=inits)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)

#Summarize the results
print('Best: %f using %s' %(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" %(mean, stdev, param))