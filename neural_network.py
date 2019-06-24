#the usual
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#keras is the most standard neural network library. it's an
#interface to tensorflow library, which by itself is a little esoteric

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adagrad,Adam
from sklearn.preprocessing import MinMaxScaler

def Neural_Network_model(number_of_inputs,number_of_classes):
    model=Sequential() #don't worry about this, every model is sequential

    #Dense is usual fully connected layers, think of it as default layer
    model.add(Dense(number_of_inputs,input_dim=number_of_inputs,kernel_initializer='normal')) #input layer
    model.add(Dense(10,activation='sigmoid')) #hidden layer, figuring out the right size is often tricky
    #can add more layers sometimes more layers  is  better, requires tuning
    model.add(Dense(number_of_classes,kernel_initializer='normal',activation='softmax')) #output layer, "softmax" means output normalized probabilities

    #here are your two most important parameters, learning rate and decay rate
    #learning rate is how fast you converge onto a minima in each step
    #decay rate is how much you decrease the learning rate each "epoch"
    #Why do you think you want to decrease the learning rate?
    optimizer=Adam(lr=0.1,decay=.01)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer)
    return model

def one_hot_encode(vector,number_of_classes):
    encoded_vector=np.zeros((len(vector),number_of_classes))
    for i in range(len(vector)):
        encoded_vector[i,vector[i]-1]=1
    return encoded_vector

def score(model,X_test,y_test):
    correct=0
    for i,point in enumerate(X_test):
        prediction=model.predict(point.reshape(1,-1))
        truth=y_test[i]

        if np.argmax(truth)==np.argmax(prediction):
            correct+=1

    print("total correct guesses "+str(correct))
    print("total data points "+str(len(y_test)))
    
wine_dataframe=pd.read_csv("wine.data")

#shuffle the dataframe (so that we get a random sample of each source
wine_dataframe = wine_dataframe.sample(frac=1).reset_index(drop=True)


#seperate data to features (X values) and labels (y values)
X=wine_dataframe.loc[:,"Alcohol":"Proline"].values #.values returns a normal matrix instead of dataframe object

#neural networks often work betterwith normalized data
scaler = MinMaxScaler()
# fit scaler on data
scaler.fit(X)
# apply transform
X = scaler.transform(X)
y=wine_dataframe["source"].values
y_encoded=one_hot_encode(y,3)

#seperate into training and testing data (say 50% train 50% test)
size_of_data=len(X)

X_train=X[:int(size_of_data*0.5)]
X_test=X[int(size_of_data*0.5):]

y_train=y_encoded[:int(size_of_data*0.5)]
y_test=y_encoded[int(size_of_data*0.5):]


#define classifier
classifier=Neural_Network_model(13,3)
print(classifier.summary())

classifier.fit(X_train,y_train,epochs=100,batch_size=16)
score(classifier,X_test,y_test)
#this network seems to perform really well
#what features is it learning?
#is it overfitting?

