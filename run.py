from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tempfile
import joblib
from google.cloud import storage, secretmanager, aiplatform
from app.config import PROJECT_ID, REGION, SECRET_NAME, PROCESSOR_DIR, PROCESSOR_FILENAME, GCS_BUCKET_NAME


def get_endpoint_id():
    """Fetch latest endpoint ID from Secret Manager"""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/{SECRET_NAME}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("utf-8")

def load_preprocessor_from_gcs(bucket_name, blob_path):
    """Download preprocessor from GCS and load it"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        blob.download_to_filename(temp_file.name)
        preprocessor = joblib.load(temp_file.name)

    return preprocessor


# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)
ENDPOINT_ID = get_endpoint_id()
endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)

PREPROCESSOR_BLOB_PATH = f"{PROCESSOR_DIR}/{PROCESSOR_FILENAME}"
preprocessor = load_preprocessor_from_gcs(GCS_BUCKET_NAME, PREPROCESSOR_BLOB_PATH)


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
        form_data["duration"] = int(form_data["duration"])
        form_data["days_left"] = int(form_data["days_left"])

        # Convert to DataFrame, then dict for Vertex AI request
        input_df = pd.DataFrame([form_data])

        transformed_input = preprocessor.transform(input_df)
        
        instance = transformed_input.tolist()

        print("input_df:", input_df)
        print("Instance for prediction:", instance)

        # Call Vertex AI endpoint
        prediction_response = endpoint.predict(instances=[instance])
        prediction = prediction_response.predictions[0]

        return f"<h2 style='text-align:center;margin-top:50px;'>Predicted Flight Price: ₹{round(prediction, 2)}</h2>"

    except Exception as e:
        return f"<h3 style='color:red;'>Error: {e}</h3>"



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

