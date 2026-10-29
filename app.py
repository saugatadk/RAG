try:
    import spaces
except ImportError:
    pass

from src.ui import create_ui

demo = create_ui()

if __name__ == "__main__":
    demo.launch()
