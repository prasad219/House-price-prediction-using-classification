from flask import Flask, render_template, request
import pickle
import numpy as np


model=pickle.load(open('model_house_pricing.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route("/predict",methods=["POST"])
def home():
    data1=request.form["area"]
    data2=request.form["bhk"]
    data3=request.form["bath"]
    data4=request.form["fur"]
    data5=request.form["loc"]
    data6=request.form["park"]
    data7=request.form["state"]
    data8=request.form["prop"]
    data9=request.form["prop1"]
    data10=request.form["sqf"]
    arr1=np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10]])
    pred=model.predict(arr1)
    return render_template("result.html",data=pred)
  

if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080)