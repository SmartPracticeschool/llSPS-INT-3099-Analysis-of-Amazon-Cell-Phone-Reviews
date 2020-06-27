# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 22:11:53 2020

@author: admin
"""


#importing libraries
import pandas as pd
import numpy as np
import matplotlib as plt
import nltk
import re

#importing the dataset
dataset=pd.read_csv(r"D:\intern_project\amazonreviews.tsv",delimiter='\t',quoting=3)


#checking for null values
print(dataset.isnull().any())

#dropping unwanted columns
dataset=dataset.drop(columns =['name', 'date','asin','helpfulVotes'],axis=1) 

l1=dataset['rating'].tolist()
new_column=[]
for i in l1:
    if i>=3:
        new_column.append(1)
    else:
        new_column.append(0)
       
#converting above list into dataframe
df=pd.DataFrame(new_column,columns=['liked_or_not'])

#adding new column to dataset
dataset['liked_or_not']=df['liked_or_not']

print(dataset.isnull().sum())

#dropping columns with null values
dataset.dropna(inplace=True)

#removing unverified reviews
dataset['verified'].replace([False],[np.nan],inplace=True)

dataset.dropna(inplace=True)

#combining title abd body columns to form a new column "review"
dataset['Review'] = dataset[['title', 'body',]].agg(' '.join, axis=1)

#dropping unwanted columns
dataset.drop(columns=['title','body','verified','rating'],inplace=True)

#selecting review column for textprocessing
x=dataset.iloc[:,1].values

# text processing

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
new_dtst=[]
for i in range(len(x)):
    review=re.sub('[^a-zA-Z]','',str(x[i]))
    review=review.lower()
    review=review.split()
    review=[word for word in review if not word in set(stopwords.words('english'))]
    p=PorterStemmer()
    review=[p.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=''.join(review)
    new_dtst.append(review)
 
#Creating "Bag of Words"
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
p=cv.fit_transform(new_dtst).toarray()
q=dataset.iloc[:,0:1].values

#splitting the data into testset and trainset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(p,q,test_size=0.2,random_state=0)

#model building

#importing libraries for building the model
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialising the model
model=Sequential()

#adding input layer to the model
model.add(Dense(units=100,init='random_uniform',activation='relu'))
#adding hidden layer
model.add(Dense(units=30,init='random_uniform',activation='relu'))
#adding output layer
model.add(Dense(units=1,activation='sigmoid'))

#configuring the learning process
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#training the model
model.fit(x_train,y_train,epochs=50,batch_size=32)

import pickle
pickle.dump(cv,open('count_vec.pkl','wb'))

#saving the model
model.save("my_model.h5")



        