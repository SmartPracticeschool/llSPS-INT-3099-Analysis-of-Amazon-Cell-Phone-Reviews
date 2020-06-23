# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 20:14:44 2020

@author: admin
"""
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


#importing dataset
dataset=pd.read_csv(r"E:\project nlp\amazonreviews.tsv",delimiter='\t',quoting=3)

#text processing
nltk.download('stopwords')
new_dtst=[]

for i in range(0,67986):
    review=re.sub('[^a-zA-Z]','',str(dataset['body'][i]))
    review=review.lower()
    review=review.split()
    review=[word for word in review if not word in set(stopwords.words('english'))]
    p=PorterStemmer()
    review=[p.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    new_dtst.append(review)
    
 #Creating "Bag of Words"
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(new_dtst).toarray()
y=dataset.iloc[:,2:3]


