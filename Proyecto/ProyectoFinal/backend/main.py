from fastapi import FastAPI, HTTPException
from generate_prediction import generate_prediction
from models import PredictionRequest, PredictionResponse

app = FastAPI(title="ChaucherApp - Priorización de Tickets")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        prediccion = generate_prediction(
            asunto=request.asunto,
            contenido=request.contenido,
            canal_ticket=request.canal_ticket,
            categoria_problema=request.categoria_problema,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al generar la predicción: {exc}") from exc

    return PredictionResponse(nivel_prioridad=prediccion)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
