from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load model & preprocessor
model = joblib.load("models/flight_price_model.pkl")
preprocessor = joblib.load("preprocessor/preprocessor.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()

        # Convert numerical fields
        form_data["duration"] = float(form_data["duration"])
        form_data["days_left"] = int(form_data["days_left"])

        # Create DataFrame (single row)
        input_df = pd.DataFrame([form_data])

        # Preprocess input
        transformed_input = preprocessor.transform(input_df)

        # Predict
        prediction = model.predict(transformed_input)[0]
        prediction = round(prediction, 2)

        return f"<h2 style='text-align:center;margin-top:50px;'>Predicted Flight Price: ₹{prediction}</h2>"

    except Exception as e:
        return f"<h3 style='color:red;'>Error: {e}</h3>"

if __name__ == "__main__":
    app.run(debug=True)
