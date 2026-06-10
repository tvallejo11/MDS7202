import pickle

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)


class WaterFeatures(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


app = FastAPI()


@app.get("/")
async def home():
    return {
        "modelo": "XGBoost optimizado con Optuna",
        "problema": "Clasificación binaria de potabilidad del agua",
        "entrada": "9 mediciones químicas: ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity",
        "salida": "potabilidad: 0 (no potable) o 1 (potable)",
    }


@app.post("/potabilidad/")
async def predict(features: WaterFeatures):
    data = [
        [
            features.ph,
            features.Hardness,
            features.Solids,
            features.Chloramines,
            features.Sulfate,
            features.Conductivity,
            features.Organic_carbon,
            features.Trihalomethanes,
            features.Turbidity,
        ]
    ]
    prediction = model.predict(data)[0]
    return {"potabilidad": int(prediction)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
