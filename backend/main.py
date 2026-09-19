from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

import numpy as np
import joblib
from pathlib import Path

from tensorflow.keras.models import load_model

app = FastAPI(
    title="Household Energy ANN API",
    description="ANN model for Global Active Power prediction",
    version="1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"

print("BASE_DIR:", BASE_DIR)
print("MODEL_DIR:", MODEL_DIR)
print("MODEL FILES:", list(MODEL_DIR.iterdir()))

model = load_model(MODEL_DIR / "ann_model.keras")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
feature_columns = joblib.load(MODEL_DIR / "feature_ columns.pkl")


class PredictionInput(BaseModel):

    global_reactive_power: float
    voltage: float

    sub_metering_1: float
    sub_metering_2: float
    sub_metering_3: float

    day: float
    month: float
    year: float
    hour: float
    minute: float


@app.get("/")
def home():

    return {"message": "Household Energy ANN API is running"}


@app.post("/predict")
def predict(data: PredictionInput):

    values = [
        data.global_reactive_power,
        data.voltage,
        data.sub_metering_1,
        data.sub_metering_2,
        data.sub_metering_3,
        data.day,
        data.month,
        data.year,
        data.hour,
        data.minute,
    ]

    input_data = np.array(values).reshape(1, -1)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled, verbose=0)

    result = float(prediction[0][0])

    return {"predicted_global_active_power": result}
