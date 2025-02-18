from flask import Flask,request,render_template

from src.pipeline.prediction_pipeline import PredictPipeline,CustomData

app=Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/predict', methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("form.html")
    else:
        print(request.form)
        data=CustomData(
            Store=request.form.get("Store"),
            Holiday_Flag=int(request.form.get("Holiday_Flag")),
            Temperature=float(request.form.get("Temperature")),
            Fuel_Price=float(request.form.get("Fuel_Price")),
            CPI=float(request.form.get("CPI")),
            Unemployment=float(request.form.get("Unemployment")),
            month=int(request.form.get("month")),
            season=request.form.get("season"),
        )

        final_data=data.get_data_as_dataframe()
        print(final_data)

        

        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_data)
        if pred is None:
                print("❌ Error: Prediction returned None!")
                return render_template("error.html", error_message="Prediction Failed. Please try again.")

        print(f"*** Final Prediction Output: {pred} ***")
        result = round(pred[0], 2)  # ✅ Safe indexing after checking for None

        return render_template("result.html", final_result=result)
    
    
        
        

        

      
        


        
    
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)
