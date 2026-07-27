import os
import cv2
import numpy as np
import logging
from pypdf import PdfReader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy loader for EasyOCR to avoid slow startups
_easyocr_reader = None

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            logger.info("Initializing EasyOCR Reader...")
            _easyocr_reader = easyocr.Reader(['en'], gpu=False) # default to CPU to avoid CUDA dependency issues
            logger.info("EasyOCR Reader initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise e
    return _easyocr_reader

def preprocess_image(image_path_or_nparray):
    """
    Applies OpenCV preprocessing to improve OCR accuracy:
    - Grayscale conversion
    - Noise removal (Bilateral filter to preserve edges)
    - Adaptive/Otsu thresholding
    """
    try:
        # Load image if path is provided
        if isinstance(image_path_or_nparray, str):
            img = cv2.imread(image_path_or_nparray)
        else:
            img = image_path_or_nparray
            
        if img is None:
            raise ValueError("Image could not be loaded")
            
        # 1. Grayscale Conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Noise Removal (Bilateral Filter preserves sharp edges while smoothing flat areas)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 3. Thresholding (Otsu's Thresholding automatically calculates threshold limit)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    except Exception as e:
        logger.warning(f"OpenCV Preprocessing failed: {e}. Using original grayscale image.")
        # Fallback to simple grayscale
        if isinstance(image_path_or_nparray, str):
            img = cv2.imread(image_path_or_nparray)
        else:
            img = image_path_or_nparray
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

def extract_text_from_image(image_path_or_nparray):
    """
    Attempts OCR using EasyOCR. Falls back to PyTesseract if EasyOCR is not available
    or fails, and falls back to friendly descriptive text if both fail.
    """
    preprocessed_img = preprocess_image(image_path_or_nparray)
    
    # Try EasyOCR
    try:
        reader = get_easyocr_reader()
        # EasyOCR can read from numpy array directly
        results = reader.readtext(preprocessed_img)
        text = "\n".join([res[1] for res in results])
        if text.strip():
            logger.info("Text successfully extracted via EasyOCR.")
            return text
    except Exception as e:
        logger.warning(f"EasyOCR failed or not installed: {e}. Trying Tesseract...")
        
    # Try PyTesseract
    try:
        import pytesseract
        # Configure common Tesseract installation path on Windows
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
                
        text = pytesseract.image_to_string(preprocessed_img)
        if text.strip():
            logger.info("Text successfully extracted via PyTesseract.")
            return text
    except Exception as e:
        logger.warning(f"PyTesseract failed or not configured: {e}")
        
    return ""

def extract_text_from_file(filepath):
    """
    Main entry point for extracting text from files (PDFs and Images).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    filename = os.path.basename(filepath).lower()
    
    # 1. Handle PDF files
    if filename.endswith('.pdf'):
        logger.info(f"Processing PDF file: {filename}")
        pdf_text = ""
        try:
            reader = PdfReader(filepath)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pdf_text += text + "\n"
            
            # If we got substantial text, return it directly
            if len(pdf_text.strip()) > 100:
                logger.info("Successfully extracted text directly from PDF.")
                return pdf_text.strip()
        except Exception as e:
            logger.warning(f"Direct PDF text extraction failed: {e}")
            
        # If direct extraction yielded very little/no text, it's likely a scanned PDF
        logger.info("Direct PDF extraction returned little/no text. Attempting PDF-to-image OCR...")
        try:
            from pdf2image import convert_from_path
            # Set poppler path dynamically if installed locally in standard directories
            pages = convert_from_path(filepath, dpi=200)
            ocr_text = ""
            for i, page in enumerate(pages):
                logger.info(f"OCR processing page {i+1}/{len(pages)}")
                # Convert PIL image to OpenCV BGR format
                open_cv_image = np.array(page)
                open_cv_image = open_cv_image[:, :, ::-1].copy() # Convert RGB to BGR
                page_text = extract_text_from_image(open_cv_image)
                ocr_text += page_text + "\n"
            if ocr_text.strip():
                return ocr_text.strip()
        except Exception as e:
            logger.error(f"PDF-to-Image OCR failed: {e}")
            
        # Final fallback: return whatever direct text we managed to extract, if any
        if pdf_text.strip():
            return pdf_text.strip()
            
        raise RuntimeError(
            "Could not extract text from PDF. The PDF appears to be a scanned document, "
            "but the Poppler library (for PDF to image conversion) or OCR engines are not available. "
            "Please upload a text-based PDF or verify that Poppler and EasyOCR/Tesseract are installed."
        )
        
    # 2. Handle Image files
    elif filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        logger.info(f"Processing Image file: {filename}")
        text = extract_text_from_image(filepath)
        if text.strip():
            return text.strip()
        raise RuntimeError(
            "Could not extract text from image. Make sure EasyOCR or Tesseract OCR is installed "
            "and that the image contains readable text."
        )
        
    else:
        raise ValueError("Unsupported file format. Only PDFs and Images (PNG, JPG, JPEG, BMP) are supported.")
