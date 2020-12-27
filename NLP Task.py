'''
Use Postman to run it. The input will be given in body with mimetype as JSON.
The method should be selected as POST.
'''



from flask import Flask,render_template,url_for,request,jsonify
import numpy as np
from transformers import pipeline
import json

zero_shot_classifier = pipeline("zero-shot-classification")
app=Flask(__name__)



# Defining the home page of our site
@app.route("/predict",methods=['POST'])  # this sets the route to this page
def predict():
    data = request.get_json()
    text = data['text']
    labels= data['labels']
    result= zero_shot_classifier(text,labels,multi_class= True)
    scores=result['scores']
    labels=result['labels']
    resultfinal=json.dumps({result['labels'][0]:result['scores'][0],result['labels'][1]:result['scores'][1],result['labels'][2]:result['scores'][2],result['labels'][3]:result['scores'][3],result['labels'][4]:result['scores'][4]})
    return resultfinal













if __name__=="__main__":
    app.run()
