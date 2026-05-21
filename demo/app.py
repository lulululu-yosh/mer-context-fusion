import gradio as gr


def predict(text: str):
    return {
        "neutral": 0.50,
        "joy": 0.15,
        "sadness": 0.10,
        "anger": 0.10,
        "surprise": 0.08,
        "fear": 0.04,
        "disgust": 0.03,
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Utterance"),
    outputs=gr.Label(label="Predicted Emotion"),
    title="MER Context Fusion Demo",
    description="Placeholder demo. Replace with trained checkpoint inference later.",
)

if __name__ == "__main__":
    demo.launch()
