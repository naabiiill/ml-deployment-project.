# Heart Disease Prediction API

A simple FastAPI application that predicts whether a record indicates heart disease. This is an educational demonstration, not a medical diagnostic tool.

## Tech stack
- FastAPI and Pydantic
- Scikit-learn Random Forest
- Joblib model serialization
- Docker and Docker Compose
- Uvicorn

## Project structure
```text
.
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── model/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── train_model.py
└── README.md
```

## Model training
The training script downloads a public 303-row Cleveland heart-disease dataset with the 13 requested input features, cleans missing values, converts the diagnosis to a binary target (0 = absence, 1 = presence), trains a Random Forest classifier, and saves it to `model/heart_model.joblib`.

Run locally (Python):
```bash
pip install -r requirements.txt
python train_model.py
uvicorn app.main:app --reload
```

## Docker
```bash
docker-compose build
docker-compose up
```
The Docker build runs `train_model.py`, which creates `model/heart_model.joblib` inside the image. Then open http://localhost:8000/docs.

## API endpoints
- `GET /health` — service and model status
- `GET /info` — model type and feature names
- `POST /predict` — returns `{"heart_disease": true}` or `false`

Example request body:
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

## Render deployment
1. Push this repository to GitHub.
2. In Render, create a new **Web Service** and connect the repository.
3. Choose **Docker** as the runtime/environment. Keep the Dockerfile at the repository root.
4. Deploy. Render will build the image and train the model during the Docker build.
5. Test `/health`, `/info`, and `/docs` using the generated Render URL.

## Notes
The dataset is used for an academic demonstration. Predictions are not medical advice and must not be used for clinical decisions.
