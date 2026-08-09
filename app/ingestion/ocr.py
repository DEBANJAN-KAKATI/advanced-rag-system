import io
from PIL import Image
from app.utils.logging import logger

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

def run_ocr_on_image_bytes(image_bytes: bytes) -> str:
    """Perform OCR on raw image bytes."""
    if not PYTESSERACT_AVAILABLE:
        logger.warning("pytesseract library not installed. Skipping OCR.")
        return ""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Basic preprocessing
        if image.mode != 'RGB':
            image = image.convert('RGB')
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        logger.warning(f"OCR processing error (Tesseract engine may not be in PATH): {e}")
        return ""
