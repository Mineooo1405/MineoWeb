from flask import Flask, render_template, request, jsonify
import pandas as pd
from joblib import load
import string

api = Flask(__name__)

@api.route('/', methods = ['GET'])
def Home():
    return render_template('index.html')

@api.route('/Naive_Bayes', methods = ['GET','POST'])
def mainFunc():
    requestType = request.method
    if requestType == 'POST':
        # Client want to 'get' the information from the backend
        #return render_template('KNN.html')
    #else:
        # Backend want to 'post' the predict value to the web for the client
        # Input recieve
        text = request.form.get('text')
        # Normalizing the input
        inputPredict = textNormalization(text)
        # Load the model 
        model = load('model/model.joblib')
        # Predict
        predict = model.predict(inputPredict)
        predictResult = 'spam' if (predict == 1) else 'non spam' 
        print (predictResult)
        return predictResult
    return render_template('Naive_Bayes.html')
    
def textNormalization(text):
    # Lowercase
    text = text.lower()
    # Remove sign
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove newline
    text = text.replace('\n', '')
    # Remove digit
    text = text.translate(str.maketrans('','', '0123456789'))
    # Split text
    words = text.split(' ')
    # Convert into pandas dataframe
    df = pd.DataFrame([words], columns=words)       
    return df

# Run and debug
if __name__ == '__main__':
    api.run(debug=True)