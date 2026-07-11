# backend/generate_prediction.py
from pathlib import Path

import cloudpickle
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

PIPELINE_PATH = Path(__file__).parent / "modelo_final.pkl"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 1024


def generate_prediction(asunto: str, contenido: str, canal_ticket: str, categoria_problema: str) -> str:
    """
    Genera la predicción de Nivel_Prioridad para un ticket nuevo, dados
    los campos mínimos necesarios (sin variables derivadas ni de leakage).
    """
    # 1. Vectorizar el texto: mismo formato usado para generar embeddings.parquet
    texto_embedding = f"Asunto_Ticket: {asunto}\nContenido_Ticket: {contenido}\n"
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        output_dimensionality=EMBEDDING_DIM,
    )
    embedding_vector = embeddings_model.embed_query(texto_embedding)

    # 2. Reconstruir el resto de las variables, idéntico al preprocesamiento de entrenamiento
    n_caracteres_ticket = len((asunto + contenido).replace("\r\n", "\n"))
    texto_bow = f"{asunto} {contenido}"

    fila = {
        "N_Caracteres_Ticket": n_caracteres_ticket,
        "Canal_Ticket": canal_ticket,
        "Categoría_Problema": categoria_problema,
        "Texto": texto_bow,
    }
    for i, valor in enumerate(embedding_vector, start=1):
        fila[f"embedding_dim_{i}"] = valor

    X_nuevo = pd.DataFrame([fila])

    # 3. Cargar el pipeline y predecir
    with open(PIPELINE_PATH, "rb") as f:
        pipeline = cloudpickle.load(f)

    prediccion = pipeline.predict(X_nuevo)[0]
    return prediccion


if __name__ == "__main__":
    resultado = generate_prediction(
        asunto="URGENTE: Transferencias no autorizadas desde mi cuenta",
        contenido=(
            "Hace 10 minutos me llegaron notificaciones de tres transferencias que "
            "yo jamas hice, por un monto muy alto. Necesito que bloqueen mi cuenta "
            "de inmediato y reviertan estos movimientos, esto es un fraude y estoy "
            "perdiendo mi dinero en este momento."
        ),
        canal_ticket="Whatsapp",
        categoria_problema="Fraude",
    )
    print("Predicción:", resultado)
