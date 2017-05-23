#Using Keras models for General Machine Learning
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy

#Function to create model required for keras classifier
def create_model():
    #create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #Compiling the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#fixing random reproducibility
seed = 7
numpy.random.seed(seed)

#Loading the dataset
dataset = numpy.loadtxt('pima-indians-diabetes.data.csv', delimiter=',')

#Splitting the data into inputs and outputs
X = dataset[:,0:8]
Y = dataset[:,8]

#Creating the model
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)

#Evaluate using 10 fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())