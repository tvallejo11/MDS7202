import gradio as gr
from services import enviar_prediccion

CANALES = ["Whatsapp", "Correo", "Página Web"]
CATEGORIAS = ["Cuenta", "Otro", "Fraude", "Técnica", "Pregunta general", "Cobros"]
TIPOS_CUENTA = ["Free", "Premium", "Business"]

TEMA = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
)

CSS_PERSONALIZADO = """
.gradio-container { max-width: 800px !important; margin: auto; }
#titulo { text-align: center; color: #4f46e5; }
"""


def predecir(asunto, contenido, canal_ticket, categoria_problema, tipo_cuenta, antiguedad_cuenta):
    # tipo_cuenta y antiguedad_cuenta se capturan para contexto, pero no se envían
    # al modelo: la Parte 2 no los seleccionó como features del pipeline final.
    if not asunto.strip() or not contenido.strip():
        return "⚠️ Por favor completa el asunto y el contenido del ticket."
    return enviar_prediccion(asunto, contenido, canal_ticket, categoria_problema)


with gr.Blocks(title="ChaucherApp - Priorización de Tickets") as demo:
    gr.Markdown("# 💜 ChaucherApp — Priorización de Tickets de Soporte", elem_id="titulo")
    gr.Markdown(
        "Completa los datos del ticket para obtener automáticamente su nivel de prioridad "
        "(**Baja**, **Media**, **Alta** o **Crítica**)."
    )

    with gr.Group():
        gr.Markdown("### 🎫 Atributos del Ticket")
        asunto = gr.Textbox(label="Asunto del ticket", placeholder="Ej: Error al iniciar sesión")
        contenido = gr.Textbox(label="Contenido del ticket", lines=5, placeholder="Describe el problema en detalle...")
        canal_ticket = gr.Dropdown(choices=CANALES, label="Canal de ingreso", value=CANALES[0])
        categoria_problema = gr.Dropdown(choices=CATEGORIAS, label="Categoría del problema", value=CATEGORIAS[0])

    with gr.Group():
        gr.Markdown("### 👤 Atributos del Usuario")
        gr.Markdown(
            "_Nota: el modelo entrenado no utiliza estos atributos en su versión actual "
            "(no fueron seleccionados como features durante el barrido de la Parte 2). "
            "Se capturan igualmente para mantener el contexto completo del ticket._"
        )
        tipo_cuenta = gr.Dropdown(choices=TIPOS_CUENTA, label="Tipo de cuenta", value=TIPOS_CUENTA[0])
        antiguedad_cuenta = gr.Slider(
            minimum=0, maximum=1200, value=180, step=1, label="Antigüedad de la cuenta (días)"
        )

    boton_predecir = gr.Button("Predecir Prioridad", variant="primary")
    resultado = gr.Textbox(label="Prioridad predicha", interactive=False)

    boton_predecir.click(
        fn=predecir,
        inputs=[asunto, contenido, canal_ticket, categoria_problema, tipo_cuenta, antiguedad_cuenta],
        outputs=resultado,
    )


if __name__ == "__main__":
    demo.launch(theme=TEMA, css=CSS_PERSONALIZADO, server_name="0.0.0.0")
