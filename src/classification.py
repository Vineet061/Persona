import json
import logging
import os
import threading
import uuid

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pytesseract
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlurredImageError(Exception):
    """Raised when an input image is too blurry to run classification/extraction on."""
    pass


modelLock = threading.Lock()


def _load_class_names():
    """Load class names from env or bundled JSON file."""
    class_names = []
    classes_json = os.environ.get("imgClasses")
    if classes_json:
        try:
            class_names = json.loads(classes_json)
        except json.JSONDecodeError:
            logger.warning("imgClasses is not valid JSON. Falling back to cls_name.json.")

    if not class_names:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls_path = os.path.join(project_root, "cls_name.json")
        if os.path.exists(cls_path):
            try:
                with open(cls_path, "r", encoding="utf-8") as f:
                    class_names = json.load(f)
            except Exception as exc:
                logger.warning(f"Failed to load cls_name.json: {exc}")

    return class_names


classes = _load_class_names()

class ModelLoader:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelLoader, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize local models only; no AWS or S3 dependency."""
        self.model = None
        self.model2 = None
        self.model3 = None
        self.model_update_weight = None
        self.ocr_reader = None
        self.classes = classes
        self._load_config()
        self._load_model()
        self._model_detection()
        self._load_ocr_reader()

    def _load_ocr_reader(self):
        """Initialize Tesseract OCR configuration for the process."""
        tesseract_path = os.environ.get("tesseractPath")
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.ocr_reader = pytesseract
        logger.info("Tesseract OCR configured")

    def get_ocr_reader(self):
        if self.ocr_reader is None:
            raise RuntimeError("OCR reader is not initialized.")
        return self.ocr_reader





    def _load_config(self):
        """Use the bundled project files directly from disk."""
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.model_local_path = os.environ.get("model") or os.path.join(self.project_root, "model_weights.onnx")
        self.updated_local_path = os.environ.get("modelNewweight") or os.path.join(self.project_root, "updated.weights.onnx")
        self.detection_local_path = os.environ.get("detectionPath") or os.path.join(self.project_root, "best.pt")
        self.localization_local_path = os.environ.get("localizationPath") or os.path.join(self.project_root, "localization.pt")

        self.classes = classes

    def _load_model(self):
        """Load the document classifier with ONNX Runtime."""
        try:
            if self.model_local_path and os.path.exists(self.model_local_path):
                logger.info(f"Loading ONNX classifier from local file: {self.model_local_path}")
                self.model = ort.InferenceSession(
                    self.model_local_path,
                    providers=["CPUExecutionProvider"],
                )
                logger.info("Base classifier loaded")
            else:
                raise FileNotFoundError(f"Classifier model not found at {self.model_local_path}")
        except Exception as exc:
            logger.error(f"Base classifier failed to load: {exc}")
            self.model = None

        try:
            if self.updated_local_path and os.path.exists(self.updated_local_path):
                current = os.path.abspath(self.model_local_path)
                candidate = os.path.abspath(self.updated_local_path)
                if candidate != current:
                    logger.info(f"Loading updated ONNX classifier from local file: {self.updated_local_path}")
                    self.model_update_weight = ort.InferenceSession(
                        self.updated_local_path,
                        providers=["CPUExecutionProvider"],
                    )
                    logger.info("Updated classifier loaded")
                else:
                    self.model_update_weight = None
            else:
                self.model_update_weight = None
        except Exception as exc:
            logger.warning(f"Updated classifier failed to load: {exc}")
            self.model_update_weight = None

        if self.model is None and self.model_update_weight is None:
            raise RuntimeError("No local classifier model could be loaded.")
        

    def _model_detection(self):
        """Load YOLO models directly from the local workspace files."""
        try:
            if self.detection_local_path and os.path.exists(self.detection_local_path):
                logger.info(f"Loading detection model from local file: {self.detection_local_path}")
                self.model2 = YOLO(self.detection_local_path)
            else:
                raise FileNotFoundError(f"Detection model not found at {self.detection_local_path}")

            if self.localization_local_path and os.path.exists(self.localization_local_path):
                logger.info(f"Loading localization model from local file: {self.localization_local_path}")
                self.model3 = YOLO(self.localization_local_path)
            else:
                raise FileNotFoundError(f"Localization model not found at {self.localization_local_path}")
        except Exception as exc:
            logger.error(f"Critical Error loading local YOLO model: {exc}")
            raise RuntimeError(f"Failed to load local model: {exc}")

    def predict(self, image):
        """
        Predicts the class of the given image.
        Args:
            image: PIL Image object
        Returns:
            tuple: (predicted_class_name, confidence_percentage)
        """
        
        if self.model_update_weight is not None:
            model= self.model_update_weight
        else:
            model = self.model
        
        if model is None:
            raise RuntimeError("Model is not initialized.")

        # Preprocess
        img = image.resize((256, 256))
        img = np.array(img)
        if img is None:
            raise ValueError(f"Could not read image: {img}")

        # Image blur check layer.
        # PIL images decode as RGB, not OpenCV's native BGR, so use COLOR_RGB2GRAY here.
        BLUR_THRESHOLD = 500
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score <= BLUR_THRESHOLD:
            # Previously this branch fell through and returned None, which crashed
            # the caller's "predictedClass, confidence = ..." unpacking. Raise a
            # clear, catchable error instead so the caller can respond gracefully.
            raise BlurredImageError(
                f"Image is too blurry to classify (blur_score={blur_score:.2f}, "
                f"threshold={BLUR_THRESHOLD})"
            )

        img_array = np.asarray(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        input_name = model.get_inputs()[0].name
        predictions = model.run(None, {input_name: img_array})[0]
        predicted_index = int(np.argmax(predictions[0]))
        if predicted_index >= len(classes):
            raise RuntimeError(
                f"Classifier returned class index {predicted_index}, "
                f"but only {len(classes)} class names are configured."
            )
        predicted_class = classes[predicted_index]
        confidence = round(100 * float(np.max(predictions[0])), 2)

        return predicted_class, confidence
        

    def modelCurrent(self):
        """
        Predicts the class of the given image.
        Args:
            image: PIL Image object
        Returns:
            tuple: (predicted_class_name, confidence_percentage)
        """
            
        if self.model_update_weight is not None:
            model= self.model_update_weight
        else:
            model = self.model
        
        if model is None:
            raise RuntimeError("Model is not initialized.")

        return model
        

    def modelCurrentDetection(self):
        """
        Predicts the class of the given image.
        Args:
            image: PIL Image object
        Returns:
            tuple: (predicted_class_name, confidence_percentage)
        """
            
        if self.model_update_weight is not None:
            model= self.model2
        else:
            model = self.model3
        
        if model is None:
            raise RuntimeError("Model is not initialized.")

            
        return model
        

        
    def predict_BB_label(self, image, request_id=None):
        if self.model2 is None:
            raise RuntimeError("Model is not initialized.")

        # Use a per-request unique filename instead of a hardcoded "1.jpg"/"imgBB/1.png" —
        # the hardcoded path was shared across every concurrent request and caused
        # one request's annotated image to overwrite another's.
        request_id = request_id or uuid.uuid4().hex

        img = image.resize(image.size)
        results = self.model2.predict(img)

        result = results[0]
        plotted_img = result.plot()

        os.makedirs("imgBB", exist_ok=True)
        out_path = os.path.join("imgBB", f"{request_id}.png")
        plt.imsave(out_path, plotted_img)
        return result, out_path
        
        

    def localization_BB(self, image):
        if self.model3 is None:
            raise RuntimeError("Model is not initialized.")

        img = image.resize(image.size) if hasattr(image, "resize") else image
        results = self.model3.predict(img)

        result = results[0]
        return result

