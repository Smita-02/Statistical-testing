from flask import Flask, render_template, request
import os
import numpy as np
from fraud_detection_project.pipeline.pipeline_prediction import PredictionPipeline

app = Flask(__name__) 

# To display home page
@app.route('/',methods=['GET'])  
def homePage():
    return render_template("index.html")

# To train the model pipeline
@app.route('/train',methods=['GET'])  
def training():
    os.system("python main.py")
    return "Done with Training" 

# To get the data from front end and pass it to pipeline for prediction
@app.route('/predict',methods=['POST','GET']) 
def index():
    if request.method == 'POST':
        try:
            #  Getting data front end home page from users
            category =float(request.form['category'])
            amt =float(request.form['amt'])
            city_pop =float(request.form['city_pop'])
            trans_month =float(request.form['trans_month'])
            trans_quarter =float(request.form['trans_quarter'])
            Cust_age =float(request.form['Cust_age'])
            city_pop_category =float(request.form['city_pop_category'])
            avg_amount_by_category =float(request.form['avg_amount_by_category'])
            gender_F =float(request.form['gender_F'])
            gender_M =float(request.form['gender_M'])
            trans_year_2019 =float(request.form['trans_year_2019'])
            trans_year_2020 =float(request.form['trans_year_2020'])
       
            # storing the user response in a numpy array format to pass it to the prediction pipeline
            data = [category,amt,city_pop,trans_month,trans_quarter,Cust_age,city_pop_category,avg_amount_by_category,
                    gender_F,gender_M,trans_year_2019,trans_year_2020]
            data = np.array(data).reshape(1, 12)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            # Returning the predicted value to the results page
            return render_template('results.html', prediction = int(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'Issue in the process', e

    else:
        return render_template('index.html')


if __name__ == "__main__":
	app.run(host="0.0.0.0", port = 8080)