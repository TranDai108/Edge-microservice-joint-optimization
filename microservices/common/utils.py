import base64, logging, os, time
from PIL import Image
import io

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def decode_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))

SERVICE_NAME = os.getenv("SERVICE_NAME", "unknown")
