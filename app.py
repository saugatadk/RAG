try:
    import spaces
except ImportError:
    pass

from src.ui import create_ui

demo = create_ui()

if __name__ == "__main__":
    # In Hugging Face Spaces (Gradio SDK), demo.launch() automatically detects
    # the environment and binds cleanly to the expected host and port.
    demo.launch()
