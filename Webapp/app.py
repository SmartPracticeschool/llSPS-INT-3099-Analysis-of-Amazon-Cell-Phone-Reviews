# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 22:37:06 2020

@author: admin
"""

import numpy as np
import pickle
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
global graph
from tensorflow.python.keras.backend import set_session
sess = tf.Session()
graph = tf.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from sklearn.feature_extraction.text import CountVectorizer

set_session(sess)


model=load_model('my_model.h5')
cv=pickle.load(open('D:\intern_project\count_vec.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/nlpmodel', methods =['GET','POST'])
def nlpmodel():
      review=request.form['Review']
      x=cv.transform([review]).toarray()
      with graph.as_default():
        y_pred=model.predict(x)
        y=y_pred[0][0]
      if y==1.0:
          
         return render_template('index.html',abc="Glad that you liked")
      else:
          return render_template('index.html',abc="Sorry for our bad performance")
    
if __name__ == '__main__':
    app.run(debug = True) 