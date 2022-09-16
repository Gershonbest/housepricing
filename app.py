
import pickle
from pydoc import render_doc 
from flask import Flask, request,app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import json
#from sklearn.preprocessing import StandardScaler


app = Flask(__name__)  # starting point of the application

# we load the pickle file as a model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/') # homepage
def home():
    return render_template('home.html')

# Creating the predict API

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict', methods=['POST'])

def predict():
    data=[float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape((1, -1)))
    print(final_input)
    output= regmodel.predict(final_input)[0]
    return render_template('home.html', prediction_text= "The house price prediction is {}".format(output))

if __name__ == '__main__':
    app.run(debug=True)