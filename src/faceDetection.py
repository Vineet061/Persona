
import cv2
import pytesseract
import os
import shutil
import uuid
import urllib.request
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename


MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
# NEGATIVE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "negative")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(NEGATIVE_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
tesseractPath="C:/Program Files/Tesseract-OCR/tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseractPath



CASCADE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cascades")
os.makedirs(CASCADE_DIR, exist_ok=True)

CASCADE_URLS = {
    "haarcascade_frontalface_default.xml":
        "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml",
    "haarcascade_eye.xml":
        "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_eye.xml",
}


def _load_cascade(name: str) -> cv2.CascadeClassifier:
    bundled = os.path.join(cv2.data.haarcascades, name) if hasattr(cv2, "data") else ""
    if bundled and os.path.isfile(bundled) and os.path.getsize(bundled) > 0:
        c = cv2.CascadeClassifier(bundled)
        if not c.empty():
            return c

    local = os.path.join(CASCADE_DIR, name)
    if not os.path.isfile(local) or os.path.getsize(local) == 0:
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(CASCADE_URLS[name], local)

    c = cv2.CascadeClassifier(local)
    if c.empty():
        raise RuntimeError(f"Failed to load cascade: {name}")
    return c


EYE_CASCADE       = _load_cascade("haarcascade_eye.xml")
EYEGLASSES_CASCADE = _load_cascade("haarcascade_eye_tree_eyeglasses.xml")
LEFTEYE_CASCADE    = _load_cascade("haarcascade_lefteye_2splits.xml")
RIGHTEYE_CASCADE   = _load_cascade("haarcascade_righteye_2splits.xml")
EYE_CASCADES = [EYE_CASCADE, EYEGLASSES_CASCADE, LEFTEYE_CASCADE, RIGHTEYE_CASCADE]




def _detect_eyes(face_gray, size=300):
    """Detect eyes in a face ROI (grayscale). Robust to small faces & glasses."""
    up = cv2.resize(face_gray, (size, size), interpolation=cv2.INTER_CUBIC)
    up = cv2.equalizeHist(up)
    upper = up[0:int(0.60 * size), :]          # eyes live in the upper 60%

    boxes = []
    for casc in EYE_CASCADES:
        for b in casc.detectMultiScale(upper, scaleFactor=1.1,
                                       minNeighbors=3, minSize=(25, 25)):
            boxes.append(tuple(int(v) for v in b))

    # distance-based NMS: drop boxes whose centers are too close (same eye)
    min_gap = size * 0.18
    kept = []
    for ex, ey, ew, eh in sorted(boxes, key=lambda e: -e[2] * e[3]):
        cx, cy = ex + ew / 2, ey + eh / 2
        if all((cx - (k[0]+k[2]/2))**2 + (cy - (k[1]+k[3]/2))**2 > min_gap**2
               for k in kept):
            kept.append((ex, ey, ew, eh))

    # a real pair: horizontally separated, vertically level
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            a, b = kept[i], kept[j]
            ax, ay = a[0]+a[2]/2, a[1]+a[3]/2
            bx, by = b[0]+b[2]/2, b[1]+b[3]/2
            if abs(ax - bx) > size * 0.18 and abs(ay - by) < size * 0.20:
                return [a, b]
    return kept   # 0 or 1 -> not a valid pair



FACE_CASCADE = _load_cascade("haarcascade_frontalface_default.xml")
EYE_CASCADE = _load_cascade("haarcascade_eye.xml")

# Thresholds
TEXT_CHAR_THRESHOLD = 40
MIN_FACE_SIZE = (45, 45)



def classify(image_path: str) -> dict:
    img = image_path
    if img is None:
        print(f"Path: {image_path}, Valid: False, Reason: Could not read image")    
        return {"path": image_path, "valid": False, "reason": "Could not read image"}

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    # 1. Face detection
    faces = FACE_CASCADE.detectMultiScale(
        gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=MIN_FACE_SIZE
    )

    if len(faces) == 0:

        print(f"Valid: False, Reason: No face detected (eyes/face not visible)")
        return {"valid": False,
                "reason": "No face detected (eyes/face not visible)"}

    # 2. ID-card check via OCR (documents contain a lot of text)
    ocr_text = pytesseract.image_to_string(gray)
    alnum_count = sum(c.isalnum() for c in ocr_text)

    if len(faces) >= 4 or alnum_count >= TEXT_CHAR_THRESHOLD:
        print(f"Valid: False, Reason: Looks like an ID card / document (faces={len(faces)}, text_chars={alnum_count})")
        return {"valid": False,
                "reason": f"Looks like an ID card / document "
                          f"(faces={len(faces)}, text_chars={alnum_count})"}

    # 3. Eye detection inside the face region
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
    face_roi = gray_eq[y:y + h, x:x + w]
    eyes = _detect_eyes(face_roi)

    if len(eyes) < 2:
        print(f"Valid: False, Reason: Face found but eyes not clearly visible (eyes_detected={len(eyes)})")
        return {"valid": False,
                "reason": f"Face found but eyes not clearly visible "
                          f"(eyes_detected={len(eyes)})"}
    
    print(f"Valid: True, Reason: Face + {len(eyes)} eyes detected, no ID-card signals")
    return {"valid": True,
            "reason": f"Face + {len(eyes)} eyes detected, no ID-card signals"}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS