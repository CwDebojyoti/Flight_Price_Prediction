Manual Deployment Steps:

Step1: Enable necessary APIs:
Activate 'Cloud Build', 'Cloud Run', 'Vertex AI', 'Artifact Registry', 'Container Registry' APIs.

Step2: Create Service Account for Vertex AI:
Create dedicated service account for Vertex AI.

Step3: Grant necessary permissions:
1. Custom Code Servic eAgent:
gcloud projects add-iam-policy-binding flight-price-prediction-470515 --member=serviceAccount:vertexai-sa@flight-price-prediction-470515.iam.gserviceaccount.com --role=roles/aiplatform.customCodeServiceAgent

2. AI Platform Admin:
gcloud projects add-iam-policy-binding flight-price-prediction-470515 --member=serviceAccount:vertexai-sa@flight-price-prediction-470515.iam.gserviceaccount.com --role=roles/aiplatform.admin

3. Storage Object Admin:
gcloud projects add-iam-policy-binding flight-price-prediction-470515 --member=serviceAccount:vertexai-sa@flight-price-prediction-470515.iam.gserviceaccount.com --role=roles/storage.objectAdmin

4. Artifact Registry Writer:
gcloud projects add-iam-policy-binding flight-price-prediction-470515 --member=serviceAccount:vertexai-sa@flight-price-prediction-470515.iam.gserviceaccount.com --role=roles/artifactregistry.writer

Step4: Build Docker Image:
docker build -t flight-price-prediction -f Dockerfile.train .

Step5: Tag the Docker Image locally:
docker tag flight-price-prediction gcr.io/flight-price-prediction-470515/flight-price-prediction

Step6: Push the image to Google Cloud Registry:
docker push gcr.io/flight-price-prediction-470515/flight-price-prediction

Step7: Submit a custom model training job manually from GCP console:
First train a new model by using custom Docker Container image. Then import the model in the Model Registry and after that create a Endpoint using this imported model. This Endpoint id can be used in the Flask app for prediction.