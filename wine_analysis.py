#first three crucial libraries

import pandas as pd #basically excel for python, allows you to build tables to work with
import numpy as np #python's fast linear algebra library, a lot of useful functions  for vector and matrix math
import matplotlib.pyplot as plt #for plotting

from scipy.optimize import curve_fit #to fit functions  to data


###############a look at data #######################
#a table of data from pandas is  called a data frame
wine_dataframe=pd.read_csv("wine.data")

print(wine_dataframe.columns)

#let's look at the first 5 rows of the frame
print(wine_dataframe.head(5))

#print a description of the data frame
print(wine_dataframe.describe())

#make a histogram of some of the features
plt.title("Distribution of Alcohol Content In Red Wine")

#show mean and standard deviation
mean=round(np.mean(wine_dataframe["Alcohol"]),1)
std=round(np.std(wine_dataframe["Alcohol"]),1)
plt.hist(wine_dataframe["Alcohol"],bins=30,label="Mean="+str(mean)+"\nstd="+str(std))
plt.legend()
plt.show() #call this function to display plots


#plot two variables against each other
plt.plot(wine_dataframe["Alcohol"],wine_dataframe["Malic_acid"],"g.")#last arguement is color (g for green) and style (. for dots)
plt.title("Alcohol Content vs. Ash")
plt.xlabel("Alcohol Content %")
plt.ylabel("Malic acid content")
plt.show()

###########regression example ###############################
#lets do linear regression between Alcohol and Malic Acid
def linear_function(x,a,b):
    return a*x+b

#fit linear function to data
results,error=curve_fit(linear_function,wine_dataframe["Alcohol"],wine_dataframe["Malic_acid"])
print("fit results")
print(results) #values for a and b
print("Fit Uncertainty")
print(error) #uncertainty on a and b (covariance matrix)

#plot linear regression results 
x=np.linspace(min(wine_dataframe["Alcohol"]),max(wine_dataframe["Alcohol"]),10)
y=linear_function(x,results[0],results[1])
plt.plot(x,y)
plt.plot(wine_dataframe["Alcohol"],wine_dataframe["Malic_acid"],"g.")
plt.show()

#######################finding good cuts#####################
#scatter matrix is kind of interesting, shows you all the correlations in one plot (gets a bit busy)
pd.plotting.scatter_matrix(wine_dataframe)
plt.show()

#we can limit it to a subsection of the dataframe
subsection_frame=wine_dataframe[["Alcohol","Malic_acid","Ash"]]
pd.plotting.scatter_matrix(subsection_frame)
plt.hist(wine_dataframe_source_one["Alcohol"],bins=30,label="source one")
plt.show()

#we can color the data  based on source (to see if there's good seperation)
plt.scatter(wine_dataframe["Alcohol"],wine_dataframe["Malic_acid"],c=wine_dataframe["source"])
plt.show()


#a histogram colored by source
wine_dataframe_source_one=wine_dataframe.loc[wine_dataframe['source'] == 1]
wine_dataframe_source_two=wine_dataframe.loc[wine_dataframe['source'] == 2]
wine_dataframe_source_three=wine_dataframe.loc[wine_dataframe['source'] == 3]
plt.hist(wine_dataframe_source_one["Alcohol"],bins=30,label="source one")
plt.hist(wine_dataframe_source_two["Alcohol"],bins=30,label="source two")
plt.hist(wine_dataframe_source_three["Alcohol"],bins=30,label="source three")
plt.legend()
plt.title("Alcohol content by source")
plt.show()
'''
TODO:: use the above tools to find variables that give you good seperation between the 3 source classes
'''
