from typing import Literal

from pydantic import BaseModel, Field

CanalTicket = Literal["Whatsapp", "Correo", "Página Web"]
CategoriaProblema = Literal["Cuenta", "Otro", "Fraude", "Técnica", "Pregunta general", "Cobros"]
NivelPrioridad = Literal["Baja", "Media", "Alta", "Critica"]


class PredictionRequest(BaseModel):
    asunto: str = Field(..., description="Asunto o título del ticket")
    contenido: str = Field(..., description="Contenido/descripción completa del ticket")
    canal_ticket: CanalTicket = Field(..., description="Canal por el que llegó el ticket")
    categoria_problema: CategoriaProblema = Field(..., description="Categoría del problema reportado")


class PredictionResponse(BaseModel):
    nivel_prioridad: NivelPrioridad
