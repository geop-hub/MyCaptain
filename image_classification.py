#importing dependcies
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#reading the csv as a dataframe
data = pd.read_csv('mnist_test.csv')

#showing the first five rows of the dataframe
data.head()

#plotting one of the rows
a=data.iloc[2,1:].values
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)

#splitting the data into features and labels
data_x=data.iloc[:,1:]
data_y=data.iloc[:,0]

#splitting the data into train and test sets
X_train,X_test,y_train,y_test=train_test_split(data_x,data_y,random_state=1,test_size=0.2)

#initialising the classifier
rfc=RandomForestClassifier(n_estimators=10)

#training the model
rfc.fit(X_train,y_train)
pred=rfc.predict(X_test)

#checking the accuracy
rfc.score(X_test,y_test)


