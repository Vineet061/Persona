"""
Flask entrypoint.

Key property of this file: nothing heavy is imported at module level. The model
stack (ONNX runtime, PyTorch via ultralytics, tesseract) is pulled in on a
background thread AFTER the HTTP socket is already listening, so the port opens
within a second of the process starting.
"""

import datetime
import json
import logging
import os
import threading
import time
from io import BytesIO

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

# Load environment variables BEFORE reading any of them.
load_dotenv()

current_datetime = datetime.datetime.now()
modelLock = threading.Lock()
imgPath = os.environ.get("imgPath")

# Was: json.loads(os.environ.get("labelList")) — a TypeError at import time if the
# env var is missing, which is the case on any host without the local .env file.
_rawLabelList = os.environ.get("labelList")
try:
    labelName = json.loads(_rawLabelList) if _rawLabelList else ["uid", "dob", "name", "gender"]
except (TypeError, ValueError):
    labelName = ["uid", "dob", "name", "gender"]

modelPath = os.environ.get("modelExtraction")
imgDummy = os.environ.get("rawImage")
imgCropped = os.environ.get("croppedImage")
labeledImgPath = os.environ.get("imgBBPath")
labeledFoldPath = os.environ.get("foldPath")
popplerPath = os.environ.get("popplerRootPath")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_FILE = "index.html"


# --------------------------------------------------------- model warmup

# _ready flips once the models are loaded. Requests arriving before then wait on
# it rather than kicking off a second concurrent load.
_ready = threading.Event()
_warmup_error = None


def _warm_models():
    """Import and initialise the model stack, off the request path."""
    global _warmup_error
    started = time.time()
    try:
        # One thread. On a fractional-CPU instance PyTorch's default thread pool
        # oversubscribes the container and inference thrashes instead of running.
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            logger.warning("Could not set torch thread count", exc_info=True)

        logger.info("Warmup: importing model stack...")
        from src.classification import ModelLoader

        logger.info("Warmup: instantiating ModelLoader...")
        ModelLoader()  # singleton — loads classifier, both YOLOs, OCR reader

        logger.info("Warmup: complete in %.1fs", time.time() - started)
    except Exception as exc:
        _warmup_error = str(exc)
        logger.exception("Warmup FAILED")
    finally:
        _ready.set()


def _require_models(timeout=300):
    """Block until warmup finishes. Returns an error string, or None if ready."""
    if not _ready.wait(timeout=timeout):
        return "Models are still loading. Try again in a moment."
    if _warmup_error:
        return "Model initialisation failed: {}".format(_warmup_error)
    return None


# ---------------------------------------------------------------- UI

@app.route("/")
def home():
    """Serve the UI from static/ or from the project root, whichever has it."""
    for folder in (os.path.join(BASE_DIR, "static"), BASE_DIR):
        if os.path.isfile(os.path.join(folder, UI_FILE)):
            resp = send_from_directory(folder, UI_FILE)
            resp.headers["Cache-Control"] = "no-store"   # always pick up edits
            return resp

    return (
        "<h1>UI file not found</h1>"
        "<p>Place <code>{f}</code> in <code>{b}</code> "
        "or in <code>{b}/static</code>.</p>".format(f=UI_FILE, b=BASE_DIR)
    ), 404


@app.route("/api/health")
def health():
    """Answers immediately, even while the models are still loading."""
    return jsonify({
        "server": "flask",
        "app": __name__,
        "modelsReady": _ready.is_set() and _warmup_error is None,
        "warmupError": _warmup_error,
        "uiFound": any(os.path.isfile(os.path.join(d, UI_FILE))
                       for d in (os.path.join(BASE_DIR, "static"), BASE_DIR)),
        "routes": sorted(str(r) for r in app.url_map.iter_rules()),
    })


# ------------------------------------------------------------ helpers

def fetch_image(img_link):
    """Download the URL and return a PIL image. Raises ValueError with a
    readable message rather than letting requests blow up mid-route."""
    try:
        response = requests.get(img_link, timeout=30)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise ValueError("The document URL took too long to respond.")
    except requests.exceptions.RequestException as exc:
        raise ValueError("Could not download the document: {}".format(exc))

    path = img_link.split("?")[0].lower()

    if path.endswith(".pdf") or response.headers.get("Content-Type", "").startswith("application/pdf"):
        from pdf2image import convert_from_bytes
        try:
            # poppler_path only applies to a local Windows install; on Linux the
            # binaries come from PATH and passing a bogus path breaks conversion.
            kwargs = {"poppler_path": popplerPath} if popplerPath and os.path.isdir(popplerPath) else {}
            pages = convert_from_bytes(response.content, **kwargs)
        except Exception:
            logger.exception("PDF conversion failed")
            raise ValueError("That PDF could not be opened. It may be corrupt "
                             "or the link may not point at a real PDF.")
        if not pages:
            raise ValueError("The PDF has no pages.")
        return pages[0].convert("RGB")

    try:
        return Image.open(BytesIO(response.content)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("The URL did not return a readable image.")


# -------------------------------------------------------------- API

@app.route("/api/verification", methods=["POST"])
def upload_datda():
    not_ready = _require_models()
    if not_ready:
        return jsonify({"status": "Failed", "errorMessage": not_ready}), 503

    # Imported here, not at module level, so startup stays cheap.
    from src.execution import execution
    from src.security import validate_url

    doc_data = {}
    startTime = datetime.datetime.utcnow()

    imgLink = request.form.get("imgLink")
    docType = request.form.get("docType") or "AADHAAR"

    if not imgLink:
        return jsonify({"status": "Failed",
                        "errorMessage": "No document URL was provided."}), 400

    if not validate_url(imgLink):
        return jsonify({"status": "Failed",
                        "errorMessage": "That URL is not valid or not allowed."}), 400

    try:
        image = fetch_image(imgLink)
    except ValueError as exc:
        return jsonify({"status": "Failed", "errorMessage": str(exc)}), 400

    _t = time.time()
    try:
        result = execution(image, startTime, doc_data, docType)
    except Exception as exc:
        logger.exception("Extraction failed")
        return jsonify({"status": "Failed", "errorMessage": str(exc)}), 500
    logger.info("[PERF] total execution: %.1fs", time.time() - _t)

    if not isinstance(result, dict):
        result = {"status": "Success", "result": result}
    result.setdefault("status", "Success")
    return jsonify(result)


# --------------------------------------------------------------- main

def _housekeeping():
    """Old-image cleanup. Runs alongside warmup instead of blocking startup."""
    try:
        from src.dataManager import docManager
        docManager()
    except Exception:
        logger.exception("docManager failed (non-fatal)")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9988))

    print("PORT env is: {}".format(os.environ.get("PORT")), flush=True)
    print("Binding to 0.0.0.0:{}".format(port), flush=True)
    print("  Health  ->  /api/health", flush=True)

    threading.Thread(target=_housekeeping, daemon=True).start()
    threading.Thread(target=_warm_models, daemon=True).start()

    # Bind immediately. Model loading continues behind this line.
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)