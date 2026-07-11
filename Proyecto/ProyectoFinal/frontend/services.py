import os

import requests
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.environ["BACKEND_URL"]


def enviar_prediccion(asunto: str, contenido: str, canal_ticket: str, categoria_problema: str) -> str:
    """
    Llama al endpoint /predict del backend y retorna el nivel de prioridad predicho,
    o un mensaje de error legible si la llamada falla.
    """
    payload = {
        "asunto": asunto,
        "contenido": contenido,
        "canal_ticket": canal_ticket,
        "categoria_problema": categoria_problema,
    }
    try:
        response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["nivel_prioridad"]
    except requests.exceptions.HTTPError:
        detalle = response.json().get("detail", "Error desconocido")
        return f"⚠️ Error de validación: {detalle}"
    except requests.exceptions.RequestException as exc:
        return f"⚠️ No se pudo contactar al servidor: {exc}"
