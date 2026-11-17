# Heart Disease Prediction API

A simple FastAPI application that predicts whether a record indicates heart disease. This is an educational demonstration, not a medical diagnostic tool.

## Live Deployment

**Render URL:** https://heart-disease-api-4cyg.onrender.com

**API Documentation (Swagger UI):** https://heart-disease-api-4cyg.onrender.com/docs

**Health Check:** https://heart-disease-api-4cyg.onrender.com/health

## Tech Stack

- FastAPI and Pydantic
- Scikit-learn Random Forest
- Joblib model serialization
- Docker and Docker Compose
- Uvicorn

## Project Structure

```text
.
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── model/
│   └── heart_model.joblib
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── train_model.py
└── README.md
```

## Model Training

The training script downloads a public 303-row Cleveland heart-disease dataset with the 13 requested input features, cleans missing values, converts the diagnosis to a binary target (0 = absence, 1 = presence), trains a Random Forest classifier, and saves it to `model/heart_model.joblib`.

### Run Locally (Python)

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs to access the API documentation.

## Docker

Build the Docker image:

```bash
docker compose build
```

Start the application:

```bash
docker compose up
```

Open http://localhost:8000/docs to access the API documentation.

The Docker build runs `train_model.py`, which creates `model/heart_model.joblib` inside the image.

## API Endpoints

- `GET /health` — service and model status
- `GET /info` — model type and feature names
- `POST /predict` — returns a heart disease prediction

### Example Request Body

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

### Example Response

```json
{
  "heart_disease": true
}
```

## Render Deployment

1. Push the repository to GitHub.
2. Create a new Web Service on Render and connect the GitHub repository.
3. Choose Docker as the runtime/environment.
4. Keep the Dockerfile at the repository root.
5. Select the Free compute plan.
6. Deploy the application.
7. Test `/health`, `/info`, and `/docs` using the generated Render URL.

Render builds the Docker image and trains the model during the Docker build.

## Notes

The dataset is used for an academic demonstration. Predictions are not medical advice and must not be used for clinical decisions.
