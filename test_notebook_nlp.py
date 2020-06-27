# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 22:33:34 2020

@author: admin
"""


from keras.models import load_model
import numpy as np
import pickle
model=load_model('my_model.h5')
corpus=[]
with open('D:\intern_project\count_vec.pkl','rb') as file:
    cv=pickle.load(file)
    inp = "the phone is awesome"
    x=cv.transform([inp])
    y=model.predict(x)
    if(y>0.5):
        print('Good review')
    else:
        print('Bad review')