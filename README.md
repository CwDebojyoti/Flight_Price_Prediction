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



Deployment using CloudBuild and establish CI/CD:

1. To Deploy the Training and Serving image and to build CI/CD, CloudBuild service has been used. Job sequence is defined in cloudbuild.yaml file. To trigger deployment using cloudbuild.yaml file the following command has been used:
'gcloud builds submit --config cloudbuild.yaml --project=flight-price-prediction-470515'

2. Once the build is successfull, it is time to build the trigger for redeployment for change in code and data. In this case push in 'git repo' branch has used as trigger for code change and 'pub/sub' notification has been used for triggering rebuild on change in data. To create the notification on Bucket the following command has been used:
'gsutil notification create -t gcs-file-changes -f json -p data/ gs://flight_price_data'


To Check how many Pub/Sub notifications are configured for the GCS bucket using:
'gsutil notification list gs://flight_price_data'

To Delete any notification:
'gsutil notification delete projects/_/buckets/flight_price_data/notificationConfigs/1'
