import os
import uvicorn
from src.api import app

if __name__ == "__main__":
    # Hugging Face Spaces routes traffic to port 7860 by default
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
