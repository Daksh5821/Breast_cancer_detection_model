from flask import Flask,request,render_template 
import pandas as pd
import numpy as np
import pickle
model=pickle.load(open("model.pkx",'rb'))
#flask app 
app = Flask(__name__)
@app.route('/')
def index():
    return render_template(('index.html'))
@app.route('/predict',methods=['POST'])
def predic():
    feature=request.form['features']
    feature_lst=feature.split(',')
    np.feature=np.asarray(feature_lst,dtype=np.float32)
    pred=model.predict(np.feature.reshape(1,-1))
    output = ["Cancerous\n" if pred[0]==1 else "Not Cancerous\n"]
    return render_template('index.html',message=output)
if __name__ == "__main__":
    app.run(debug=True)