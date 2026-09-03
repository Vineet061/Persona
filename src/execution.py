from src.classification import *

import threading, os, uuid, time
import pytesseract
from PIL import Image
import cv2
from dotenv import load_dotenv
import datetime


load_dotenv()
modelLock = threading.Lock()
imgPath = os.environ.get("imgPath")

# Was: json.loads(os.environ.get("labelList")) — raises TypeError at import time
# whenever the env var is absent, which is the case on any host without the local
# .env file. Fall back to the known label order instead.
_rawLabelList = os.environ.get("labelList")
try:
    labelName = json.loads(_rawLabelList) if _rawLabelList else ["uid", "dob", "name", "gender"]
except (TypeError, ValueError):
    labelName = ["uid", "dob", "name", "gender"]

imgDummy = os.environ.get("rawImage")
imgCropped = os.environ.get("croppedImage")
tesseractLoc = os.environ.get("tesseractPath")
labeledImgPath = os.environ.get("imgBBPath")
labeledFoldPath = os.environ.get("foldPath") or "imgBB/"
portNumber = os.environ.get("portNum")

# Only override the tesseract binary if the configured path actually exists. The
# value in .env is a Windows path and meaningless on a Linux container — there we
# want whatever "tesseract" is on PATH.
if tesseractLoc and os.path.exists(tesseractLoc):
    pytesseract.pytesseract.tesseract_cmd = tesseractLoc

current_datetime = datetime.datetime.now()

# Largest edge we feed to the models. Full-resolution phone photos are several
# thousand pixels wide and cost far more CPU than they add in accuracy.
MAX_INPUT_SIDE = 1600


def extract_look_pytessrect(imgCropped):
    try:
        extractedText = pytesseract.image_to_string(imgCropped)
        lines = [line.strip() for line in extractedText.split('\n') if line.strip()]
        return lines
    except Exception as e:
        return f"Error in extract_look_pytessrect: {str(e)}"


def _downscale(img):
    """Cap the longest edge at MAX_INPUT_SIDE, preserving aspect ratio."""
    if max(img.size) <= MAX_INPUT_SIDE:
        return img
    ratio = MAX_INPUT_SIDE / max(img.size)
    newSize = (int(img.width * ratio), int(img.height * ratio))
    print(f"[PERF] downscaling input {img.size} -> {newSize}", flush=True)
    return img.resize(newSize, Image.LANCZOS)


def detect_rotation_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)      # Hough Transform used to detect lines in the image, which help in determining the orientation in the docx.
    # rho- Distance from origin, theta - angle of the line.

    angles = []

    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)

    if len(angles) == 0:
        return 0

    return np.median(angles)


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Fix cropping
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(image, M, (new_w, new_h))


def correct_orientation(image, doc_data):
    angle = detect_rotation_angle(image)
    if abs(angle) > 10:
        print(f"Detected angle: {angle}")
        doc_data["img_rotation_angle"] = int(angle)

        if angle == 0:
            h, w = image.shape[:2]
            if h > w:
                print("90 degree")
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif w > h or w == h:
                print("0")

        corrected = rotate_image(image, angle)

        return corrected
    else:
        print("The angle of image is", angle)
        return image


def execution(inputImg, startingTime, doc_data, docType):
    json_data2 = {}
    json_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not os.path.exists("json_data"):
        os.makedirs("json_data")
    os.makedirs(labeledFoldPath, exist_ok=True)

    request_id = uuid.uuid4().hex

    inputImg = _downscale(inputImg)

    model_loader = ModelLoader()

    _t = time.time()
    try:
        predictedClass, confidence = model_loader.predict(inputImg)
    except BlurredImageError as e:
        return ({
            "status": "Failed",
            "type": "UNKNOWN",
            "message": "The image is too blurry to process. Please retake a clearer photo and try again.",
            "errorMessage": str(e),
        })
    print(f"[PERF] classify: {time.time() - _t:.1f}s", flush=True)

    valProcessingTime = datetime.datetime.utcnow()
    valTime = (valProcessingTime - startingTime).total_seconds()
    print(predictedClass, confidence, "This is the answer")

    doc_data["predicted_class"] = predictedClass
    doc_data["confidence"] = str(confidence)

    # predicted class and docType comp here
    doc_data["validationResult"] = (doc_data["predicted_class"] == docType)

    print(doc_data["validationResult"], "This is the validation result", flush=True)
    if doc_data["validationResult"] is True:

        _t = time.time()
        imgSaveResult = model_loader.localization_BB(inputImg)
        print(f"[PERF] localization_BB: {time.time() - _t:.1f}s", flush=True)

        labelBB = imgSaveResult.boxes.xyxy.tolist()
        labelData = imgSaveResult.boxes.cls.tolist()
        labelConf = imgSaveResult.boxes.conf.tolist()

        if len(labelBB) != 0:
            cropped_image = inputImg.crop(labelBB[0])
            cropped_image.save("cropped_image.jpg")
            image = cv2.imread("cropped_image.jpg")
            corrected = correct_orientation(image, json_data2)
            cv2.imwrite("output.jpg", corrected)
        else:
            inputImg.save("cropped_image.jpg")
            image = cv2.imread("cropped_image.jpg")
            corrected = correct_orientation(image, json_data2)
            cv2.imwrite("output.jpg", corrected)
        #---------------------------------------------------------------------------
        if predictedClass == "AADHAAR":
            img_current = Image.open("output.jpg")

            _t = time.time()
            imgSave, bbImgPath = model_loader.predict_BB_label(img_current, request_id)
            print(f"[PERF] predict_BB_label: {time.time() - _t:.1f}s", flush=True)

            imageResized = img_current

            labelBB = imgSave.boxes.xyxy.tolist()
            labelData = imgSave.boxes.cls.tolist()
            labelConf = imgSave.boxes.conf.tolist()

            if len(labelConf) > 0:
                labelList = list(map(int, labelData))

                unique_values_with_highest_accuracy = {}
                for value, accuracy, listBB in zip(labelList, labelConf, labelBB):
                    if value not in unique_values_with_highest_accuracy or accuracy > unique_values_with_highest_accuracy[value]['accuracy']:
                        unique_values_with_highest_accuracy[value] = {'accuracy': accuracy, 'l': listBB}

                sorted_result = sorted(unique_values_with_highest_accuracy.items(), key=lambda x: x[1]['accuracy'],
                                        reverse=True)

                result = [value for value, _ in sorted_result]
                conf = [data['accuracy'] for _, data in sorted_result]
                values = [data['l'] for _, data in sorted_result]

                mainLt = result
                mainConf = conf
                mainBB = values

                jsonData = {}
                if not mainLt:
                    print(f"[EXTRACTION] Failed to detect any labels for {predictedClass}. mainLt={mainLt}")
                    return({"message": "Model couldn't able to extract the text.", "type": predictedClass})

                _t = time.time()
                for m_range in range(len(mainLt)):
                    classNameInt = int(mainLt[m_range])

                    if classNameInt < 0 or classNameInt >= len(labelName):
                        print(f"[EXTRACTION] Skipping unmapped class id: {classNameInt}. labelName={labelName}")
                        continue

                    jKey = labelName[classNameInt]

                    conf_key = mainConf[m_range]
                    json_data2[jKey + " detection-confidence"] = conf_key

                    cropped_image = imageResized.crop(mainBB[m_range])

                    reader = model_loader.get_ocr_reader()
                    field_text = reader.image_to_string(np.array(cropped_image), config="--psm 6")
                    if not field_text.strip():
                        print(f"[EXTRACTION][{jKey}] field crop OCR empty, falling back to full image OCR")
                        field_text = reader.image_to_string(np.array(imageResized), config="--psm 6")

                    result = [line.strip() for line in field_text.splitlines() if line.strip()]
                    print(f"[EXTRACTION][{jKey}] raw OCR result: {result}")
                    final_data = " ".join(result)
                    if final_data.strip():
                        jsonData[jKey] = final_data
                print(f"[PERF] ocr ({len(mainLt)} fields): {time.time() - _t:.1f}s", flush=True)

                if not jsonData:
                    print(f"[EXTRACTION] No valid mapped OCR fields extracted. mainLt={mainLt}, labelName={labelName}")
                    full_text = model_loader.get_ocr_reader().image_to_string(np.array(imageResized), config="--psm 6")
                    print(f"[EXTRACTION] Full-image OCR fallback text: {full_text}")
                    return({"message": "Model couldn't able to extract the text.", "type": predictedClass})

                print(f"[EXTRACTION] extracted jsonData before post-processing: {jsonData}")

                # gender
                if "gender" in jsonData.keys():
                    genderDetail = jsonData["gender"]
                    if len(genderDetail) > 5:
                        jsonData["gender"] = "female"
                    elif len(genderDetail) < 5:
                        jsonData["gender"] = "male"

                if "dob" in jsonData.keys():
                    dobDetail = jsonData["dob"]

                    if len(dobDetail) < 6:
                        jsonData["dob"] = dobDetail
                    else:
                        try:
                            dobYear = dobDetail[-4:]
                            first_num = dobYear[0]
                            if first_num in ("8", "6", "3"):
                                dobYear[0] = "2"
                            elif first_num in ("4", "7"):
                                dobYear[0] == "1"
                            dobDate = dobDetail[0:2]

                            if (dobDetail[2] == "/") or (dobDetail[2] == "1"):
                                dobMonth = dobDetail[3:5]
                            elif dobDetail[2] == "0":
                                dobMonth = dobDetail[2:4]
                            else:
                                dobMonth = "09"

                            final_date = dobDate + "/" + dobMonth + "/" + dobYear
                            jsonData["dob"] = final_date
                        except Exception as e:
                            # Don't let an unexpected OCR format crash the whole
                            # extraction — keep the raw OCR'd value instead.
                            print(f"DOB post-processing failed, keeping raw OCR value: {e}")

                if "uid" in jsonData:
                    uid = jsonData["uid"]
                    newUid = "".join(num for num in uid if num != " ")
                    if len(uid) < 12:
                        jsonData["remark"] = "detect uid does not have total 12 numbers"
                    jsonData["uid"] = newUid
                else:
                    # The localization model didn't detect a uid field on this card —
                    # this used to raise an unhandled KeyError further down.
                    jsonData["uid"] = ""
                    jsonData["remark"] = "Could not extract the ID number from the document"

                timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

                finalUid = jsonData.get("uid") or "unidentified"
                finalImageName = f"{finalUid}_{timestamp}.png"

                try:
                    if os.path.exists(bbImgPath):
                        os.rename(bbImgPath, os.path.join(labeledFoldPath, finalImageName))
                except OSError as e:
                    print(f"Could not persist annotated detection image: {e}")

                extractionProcessingTime = datetime.datetime.utcnow()
                extractionTimeTaken = (extractionProcessingTime - valProcessingTime).total_seconds() * 100
                totalTime = extractionTimeTaken + valTime
                mainJson = {}
                mainJson["totalProcessTime"] = int(totalTime)

                jsonData["processDuration"] = int(extractionTimeTaken * 1000)
                mainJson["confidence"] = str(confidence)
                mainJson["processDuration"] = int(valTime * 1000)
                mainJson["data"] = jsonData
                mainJson["type"] = predictedClass
                mainJson["validationResult"] = True
                print(doc_data, mainJson, "this is the doc_data")

                with open("json_data/" + str(json_filename) + ".json", 'w') as json_file:
                    json.dump(json_data2, json_file, indent=4)
                print(f"[EXTRACTION] final payload: {mainJson}", flush=True)
                return ({"docDetail": mainJson, "type": predictedClass, "status": "Success", "message": "Successfully identified & extracted."})
            else:
                mainJson = {}
                mainJson["type"] = predictedClass
                mainJson["confidence"] = str(confidence)
                mainJson["processDuration"] = int(valTime * 1000)
                mainJson["validationResult"] = True

                with open("json_data/" + str(json_filename) + ".json", 'w') as json_file:
                    json.dump(json_data2, json_file, indent=4)
                print(f"[EXTRACTION] extraction failed payload: {mainJson}", flush=True)
                return ({"docDetail": mainJson, "status": "failed", "message": "Identification successful, extraction failed. Model could not able to locate the id number"})

        elif predictedClass == "UNKNOWN":
            validationTime = int(valTime * 1000)

            startingTime = str(startingTime)
            with open("json_data/" + str(json_filename) + ".json", 'w') as json_file:
                json.dump(json_data2, json_file, indent=4)

            return ({"status": "Success", "type": "unidentified", "confidence": str(confidence), "processDuration": validationTime, "message": "Identification successful", "validationResult": True})
        else:
            validationTime = int(valTime * 1000)
            with open("json_data/" + str(json_filename) + ".json", 'w') as json_file:
                json.dump(json_data2, json_file, indent=4)
            return ({"status": "Success", "type": predictedClass, "confidence": str(confidence), "processDuration": validationTime, "message": "Identification successful", "validationResult": True})

    else:
        return ({"status": "Success", "type": doc_data["predicted_class"], "confidence": doc_data["confidence"], "message": "Identification successful", "validationResult": False})