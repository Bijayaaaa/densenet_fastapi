import gradio as gr
from app.model_loader import predict_from_bytes, load_model_once
import io

load_model_once()

def predict_gradio(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    result = predict_from_bytes(buf.read())
    return result.get("predicted_class", "N/A"), result.get("confidence", 0.0)

iface = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(label="Predicted Class"), gr.Number(label="Confidence")],
    title="DenseNet TinyImageNet"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)
