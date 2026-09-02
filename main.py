# from src.classification import *
# from src.execution import *
# from src.dataManager import *
# from src.security import validate_url

# import logging
# import json
# import os
# import threading
# import datetime
# from io import BytesIO

# import requests
# import pytesseract
# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# from dotenv import load_dotenv
# from PIL import Image, UnidentifiedImageError
# from pdf2image import convert_from_bytes

# # Local imports

# # Load environment variables BEFORE reading any of them.
# load_dotenv()

# current_datetime = datetime.datetime.now()
# modelLock = threading.Lock()
# imgPath = os.environ.get("imgPath")
# labelName = json.loads(os.environ.get("labelList"))
# modelPath = os.environ.get("modelExtraction")
# imgDummy = os.environ.get("rawImage")
# imgCropped = os.environ.get("croppedImage")
# labeledImgPath = os.environ.get("imgBBPath")
# labeledFoldPath = os.environ.get("foldPath")
# popplerPath = os.environ.get("popplerRootPath")

# logging.basicConfig(level=logging.INFO,
#                     format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# app = Flask(__name__)
# CORS(app)

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# UI_FILE = "index.html"


# # ---------------------------------------------------------------- UI

# @app.route("/")
# def home():
#     """Serve the UI from static/ or from the project root, whichever has it."""
#     for folder in (os.path.join(BASE_DIR, "static"), BASE_DIR):
#         if os.path.isfile(os.path.join(folder, UI_FILE)):
#             resp = send_from_directory(folder, UI_FILE)
#             resp.headers["Cache-Control"] = "no-store"   # always pick up edits
#             return resp

#     return (
#         "<h1>UI file not found</h1>"
#         "<p>Place <code>{f}</code> in <code>{b}</code> "
#         "or in <code>{b}\\static</code>.</p>".format(f=UI_FILE, b=BASE_DIR)
#     ), 404


# @app.route("/api/health")
# def health():
#     """Quick check that you are talking to Flask and not another server."""
#     return jsonify({
#         "server": "flask",
#         "app": __name__,
#         "uiFound": any(os.path.isfile(os.path.join(d, UI_FILE))
#                        for d in (os.path.join(BASE_DIR, "static"), BASE_DIR)),
#         "routes": sorted(str(r) for r in app.url_map.iter_rules()),
#     })


# # ------------------------------------------------------------ helpers

# def fetch_image(img_link):
#     """Download the URL and return a PIL image. Raises ValueError with a
#     readable message rather than letting requests blow up mid-route."""
#     try:
#         response = requests.get(img_link, timeout=30)
#         response.raise_for_status()
#     except requests.exceptions.Timeout:
#         raise ValueError("The document URL took too long to respond.")
#     except requests.exceptions.RequestException as exc:
#         raise ValueError("Could not download the document: {}".format(exc))

#     path = img_link.split("?")[0].lower()

#     if path.endswith(".pdf") or response.headers.get("Content-Type", "").startswith("application/pdf"):
#         try:
#             pages = convert_from_bytes(response.content, poppler_path=popplerPath)
#         except Exception as exc:
#             logger.exception("PDF conversion failed")
#             raise ValueError("That PDF could not be opened. It may be corrupt "
#                              "or the link may not point at a real PDF.")
#         if not pages:
#             raise ValueError("The PDF has no pages.")
#         return pages[0].convert("RGB")

#     try:
#         return Image.open(BytesIO(response.content)).convert("RGB")
#     except UnidentifiedImageError:
#         raise ValueError("The URL did not return a readable image.")


# # -------------------------------------------------------------- API

# @app.route("/api/verification", methods=["POST"])
# def upload_datda():
#     doc_data = {}
#     startTime = datetime.datetime.utcnow()

#     imgLink = request.form.get("imgLink")
#     docType = request.form.get("docType") or "AADHAAR"

#     if not imgLink:
#         return jsonify({"status": "Failed",
#                         "errorMessage": "No document URL was provided."}), 400

#     if not validate_url(imgLink):
#         return jsonify({"status": "Failed",
#                         "errorMessage": "That URL is not valid or not allowed."}), 400

#     try:
#         image = fetch_image(imgLink)
#     except ValueError as exc:
#         return jsonify({"status": "Failed", "errorMessage": str(exc)}), 400

#     try:
#         result = execution(image, startTime, doc_data, docType)
#     except Exception as exc:
#         logger.exception("Extraction failed")
#         return jsonify({"status": "Failed", "errorMessage": str(exc)}), 500

#     if not isinstance(result, dict):
#         result = {"status": "Success", "result": result}
#     result.setdefault("status", "Success")
#     return jsonify(result)


# if __name__ == "__main__":
#     # docManager()
#     # port = int(os.environ.get("PORT") or 9988)
#     app.run(host="0.0.0.0",port=9988)


import os
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "alive"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9988))
    print("PORT env is:", os.environ.get("PORT"), flush=True)
    print("binding to 0.0.0.0:", port, flush=True)
    app.run(host="0.0.0.0", port=port)