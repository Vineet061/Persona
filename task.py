
from src.classification import *
from src.execution import *
from src.dataManager import *
from src.security import validate_url

from pathlib import Path
import requests
from io import BytesIO
from pdf2image import convert_from_bytes
from PIL import Image, UnidentifiedImageError
import os

popplerPath = os.environ.get("popplerRootPath")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")



import json



# @celery.task(name="task.Extraction",bind=True) 
def Extraction(self,imgLink,startTime,memberId,docId,doc_data,docType):
    validationResult = None
    if not imgLink:
        return ({'error': 'url is not provided'})
    
    if not memberId:
        return ({'error': 'Member Id is not provided'})
    
    if not docType:
        return ({'error': 'Document Type not provided'})
    
    if not validate_url(imgLink):
        return ({'status': 'Failed', 'error': 'Invalid or unsafe image URL provided'})
    
    
    if not os.path.exists("json_data"):
        os.makedirs("json_data")

    # if (docType=="pdf") or (".pdf" in imgLink):
    if ".pdf" in imgLink:
        response = requests.get(imgLink, timeout=10)
        pdfBytes = response.content

        # Convert PDF pages to images
        try:
            images = convert_from_bytes(pdfBytes, poppler_path=popplerPath)
            image = images[0]
        except Exception as e:
            return ({"status":"Failed","message":"Provided link is not correct. It might has hex string"})    
    else:
        print("imgggg")
        response = requests.get(imgLink)
        try:
            print("12345")
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except UnidentifiedImageError as e:
            print("Coming hererererer")
            return ({"status":"Failed","errorMessage": str(e)}), 500

    try:
        print("executionnn",image)
        val = execution(image,startTime,doc_data,docType)
        # os.remove("cropped_image.jpg")
        # os.remove("output.jpg")


        print(val,"This is the val")
        print(val["status"],"sdfdsf")
        print(val["docDetail"],"sdrsdfsdfdsfsdfdsf")
        print(val["docDetail"]["type"],"checkkkkkkkkkk")

        if (val["docDetail"]["type"]=="AADHAAR"):
            
            payload = {
                "memberId": int(memberId),
                "extractedData": val['docDetail'],
                "documentUrl": imgLink,
                "documentId":int(docId),
            }
            print(payload,"This is payload")

            try:                
                external_api_response = requests.post(
                    "https://staging.basicxsports.in/api/v1/document-extractions",
                    json=payload,
                    timeout=15
                    )
                   
                if external_api_response.status_code not in [200, 201]:
                    return (
                        (
                            {
                                "error": "Failed to save document to external API",
                                "status_code": external_api_response.status_code,
                                "response": external_api_response.text,
                            }
                        ),
                        500,
                    )

                return ({"message": "Document classification successful"}), 200

            except requests.RequestException as api_error:
                print("fail to connecttt")
                return (
                    (
                        {
                            "error": "Failed to connect to external API",
                            "details": str(api_error),
                        }
                    ),
                    500,
                )
        
        elif (val["type"]=="unidentified"):
            payload = {
                "memberId": int(memberId),
                "extractedData": val,
                "documentUrl": imgLink,
                "documentId":int(docId),
            }
            print(payload,"This is payload")

            try:
                external_api_response = requests.post(
                    "https://staging.basicxsports.in/api/v1/document-extractions",
                    json=payload,
                    timeout=15,
                    
                )

                print(external_api_response,"This is check 2")
                if external_api_response.status_code not in [200, 201]:
                    return (
                        (
                            {
                                "error": "Failed to save document to external API",
                                "status_code": external_api_response.status_code,
                                "response": external_api_response.text,
                            }
                        ),
                        500,
                    )

                return ({"message": "Document classification successful"}), 200

            except requests.RequestException as api_error:
                print("fail to connecttt")
                return (
                    (
                        {
                            "error": "Failed to connect to external API",
                            "details": str(api_error),
                        }
                    ),
                    500,
                )
            
        else:
            payload = {
                "memberId": int(memberId),
                "extractedData": val,
                "documentUrl": imgLink,
                "documentId":int(docId)
            }


    except Exception as e:
        return ({"status":"Failed","errorMessage": str(e)}),500
    



# @celery.task(name="task.faceDetect") 
def faceDetect(img):
    if not validate_url(img):
        return ("invalid_url")
    
    response = requests.get(img)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    # if True:
    image = np.array(image)
    result = classify(image)
    if result["valid"] ==  False:
        return("fail")
    else:
        return("pass")
        
        




# @celery.task(name="task.scoreSheet",bind=True) 
# def scoreSheet(self,imgLink,uid):
#     validationResult = None
#     if not imgLink:
#         return ({'error': 'url is not provided'})
    
#     if not uid:
#         return ({'error': 'Member Id is not provided'})
   
    
#     if not validate_url(imgLink):
#         return ({'status': 'Failed', 'error': 'Invalid or unsafe image URL provided'})
    
    
#     if not os.path.exists("json_data"):
#         os.makedirs("json_data")

#     response = requests.get(imgLink, timeout=10)
#     pdfBytes = response.content

#     with tempfile.TemporaryDirectory() as tmp_dir:
#         tmp_path = os.path.join(tmp_dir, f"{uid}.pdf")

#         # Save original PDF
#         with open(tmp_path, "wb") as f:
#             f.write(pdfBytes)

#         # Convert PDF to images if needed
#         images = convert_from_bytes(pdfBytes, poppler_path="poppler-23.11.0/Library/bin")
#         upload = images[0]

#         # Use pdf_path here
#         print(tmp_path)

#         try:
#             state = run(tmp_path, json_path=None)
#         except Exception as exc:
#             return jsonify({"error": f"extraction failed: {exc}"}), 422

#     payload = state["json_output"]
#     payload["sourcePdf"] = uid+".pdf"  # report the original name, not the temp path

#     # persist the same round/PQF/QF/SF/Finals-grouped structure returned
#     # below (see json_export.build_export) rather than just the response
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     json_path = os.path.join(OUTPUT_DIR, Path(uid+".pdf").stem + ".json")
#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(payload, f, indent=2, ensure_ascii=False)
#     print(payload,"Thisis the payload")

#     try:
#         result = {
#                 "fileId": uid,
#                 "data": payload,
#             }
#     except Exception as e:
#         result = {
#             "fileId": uid,
#             "error": str(e),
#         }

#     publish_result(result)
#     return (payload)





# @celery.task(name="task.scoreSheet",bind=True) 
def scoreSheet(imgLink,uid):
    validationResult = None
    if not imgLink:
        return ({'error': 'url is not provided'})
    
    if not uid:
        return ({'error': 'Member Id is not provided'})
   
    
    if not validate_url(imgLink):
        return ({'status': 'Failed', 'error': 'Invalid or unsafe image URL provided'})
    
    
    if not os.path.exists("json_data"):
        os.makedirs("json_data")

    response = requests.get(imgLink, timeout=10)
    pdfBytes = response.content

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, f"{uid}.pdf")

        # Save original PDF
        with open(tmp_path, "wb") as f:
            f.write(pdfBytes)

        # Convert PDF to images if needed
        images = convert_from_bytes(pdfBytes, poppler_path="poppler-23.11.0/Library/bin")
        upload = images[0]

        # Use pdf_path here
        print(tmp_path)

        try:
            state = run(tmp_path, json_path=None)
        except Exception as exc:
            result = {
                "success": False,
                "fileId": uid,
                "error": str(exc),
            }

            publish_result(result)
            return result

    payload = state["json_output"]
    payload["sourcePdf"] = uid+".pdf"  # report the original name, not the temp path

    # persist the same round/PQF/QF/SF/Finals-grouped structure returned
    # below (see json_export.build_export) rather than just the response
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, Path(uid+".pdf").stem + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(payload,"Thisis the payload")

    try:
        result = {
                "success": True,
                "fileId": uid,
                "data": payload,
            }
    except Exception as e:
        result = {
            "success": False,
            "fileId": uid,
            "error": str(e),
        }

    return(result)
    # return (payload)

















