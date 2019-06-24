#goal is to correctly classify the source of the wine based on properties
#source is location in italy, labeled location 1,2 or 3

from sklearn.tree import DecisionTreeClassifier as DecisionTree
import pandas as pd #basically excel for python, allows you to build tables to work with
import numpy as np #python's fast linear algebra library, a lot of useful functions  for vector and matrix math
import matplotlib.pyplot as plt #for plotting
import os
from  sklearn.model_selection import GridSearchCV,RandomizedSearchCV # model hyperparameter selection
from sklearn.tree import export_graphviz
import pickle #ways to save/reload python objects

##################building a decision tree classifier###############
#our data
wine_dataframe=pd.read_csv("wine.data")

#define classifier
classifier=DecisionTree()

#shuffle the dataframe (so that we get a random sample of each source
wine_dataframe = wine_dataframe.sample(frac=1).reset_index(drop=True)


#seperate data to features (X values) and labels (y values)
X=wine_dataframe.loc[:,"Alcohol":"Proline"].values #.values returns a normal matrix instead of dataframe object
y=wine_dataframe["source"].values

#seperate into training and testing data (say 50% train 50% test)
size_of_data=len(X)
X_train=X[:int(size_of_data*0.5)]
X_test=X[int(size_of_data*0.5):]

y_train=y[:int(size_of_data*0.5)]
y_test=y[int(size_of_data*0.5):]


#fit the tree
classifier.fit(X_train,y_train)

#check how well it performs on test set
print(classifier.score(X_test,y_test))

#an interesting way to visiualize what the tree came up with 
export_graphviz(classifier,out_file="tree.dot")

#not sure this works on windows, try and let me know
os.system("dot -Tpng tree.dot -o tree.png")
#open up the tree.png file

#save tree
saved_file=open("Wine_Decision_Tree.pkl","wb")
pickle.dump(classifier,saved_file,pickle.HIGHEST_PROTOCOL)
#load it like this classifier=pickle.load("Wine_Decision_Tree.pkl")


########################hyper parameter search######################
'''
a hyper parameter is some parameter of the classifier that we set by hand. 
for decision trees you can check the parameters here
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
we'll use RandomizedSearchCV to search for the best hyper parameters. RandomizedSearchCV searchs around the parameter space randomly
while GridSearchCV exhaustivly checks every possible value. In practice a random search works nearly as well and much faster
'''

pars_to_search={"criterion":["gini","entropy"],"max_depth":[i for i in range(1,10)],"min_samples_split":[i for i in range(2,50)]}
clf=DecisionTree()
search=RandomizedSearchCV(clf,pars_to_search,cv=5,iid=False,refit=True)
search.fit(X_train,y_train)
best_tree=search.best_estimator_

export_graphviz(best_tree,out_file="best_tree.dot")

#not sure this works on windows, try and let me know
os.system("dot -Tpng best_tree.dot -o best_tree.png")
#open up the best_tree.png file
print(best_tree.score(X_test,y_test))
