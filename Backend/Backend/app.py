import os
import uuid
import shutil
import math

import cv2
import numpy as np
import gradio as gr

from PIL import Image
from ultralytics import YOLO

from fastapi import (
    FastAPI,
    UploadFile,
    File
)

from fastapi.responses import (
    JSONResponse,
    FileResponse
)

from fastapi.middleware.cors import (
    CORSMiddleware
)

# ==========================================================
# PROJECT CONFIGURATION
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best.pt"
)

UPLOAD_FOLDER = "/tmp/uploads"
RESULT_FOLDER = "/tmp/results"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)

# ==========================================================
# LOAD YOLO MODEL
# ==========================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

print("\nLoading YOLO Model...")

model = YOLO(MODEL_PATH)

print("YOLO Model Loaded Successfully.")

# ==========================================================
# DETECTION SETTINGS
# ==========================================================

# Main confidence threshold
CONF_THRESHOLD = 0.42

# Image size for inference
IMAGE_SIZE = 1280

# Duplicate suppression
IOU_THRESHOLD = 0.70

# Maximum detections
MAX_DETECTIONS = 1000

# Minimum object dimensions
MIN_WIDTH = 8
MIN_HEIGHT = 8

# Area limits
MIN_BOX_AREA = 0.00002
MAX_BOX_AREA = 0.65

# Shape limits
MIN_ASPECT_RATIO = 0.18
MAX_ASPECT_RATIO = 5.50

# Border rejection
BORDER_MARGIN = 3

# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

ENABLE_PREPROCESSING = True

ENABLE_CLAHE = True

ENABLE_SHARPEN = True

ENABLE_GAMMA = False

GAMMA_VALUE = 1.15

CLAHE_CLIP = 2.5

CLAHE_GRID = (8, 8)

# ==========================================================
# DETECTION IMPROVEMENTS
# ==========================================================

ENABLE_DUPLICATE_REMOVAL = True

ENABLE_BORDER_FILTER = True

ENABLE_AREA_FILTER = True

ENABLE_ASPECT_FILTER = True

ENABLE_SMALL_OBJECT_FILTER = True

ENABLE_SECOND_PASS_FILTER = True

# ==========================================================
# API SETTINGS
# ==========================================================

API_VERSION = "3.0"

PROJECT_NAME = "AI Bolt & Nut Detector"

SUPPORTED_CLASSES = [
    "bolt",
    "nut"
]

# ==========================================================
# GRADIO PREDICTION
# ==========================================================

def predict_gradio(image):

    if image is None:
        return None, "No image selected."

    unique_id = str(uuid.uuid4())[:8]

    input_filename = f"input_{unique_id}.jpg"

    input_path = os.path.join(
        UPLOAD_FOLDER,
        input_filename
    )

    image.save(input_path)

    # ------------------------------------------------------
    # Image Preprocessing
    # ------------------------------------------------------

    if ENABLE_PREPROCESSING:
        preprocess_image(input_path)

    img = cv2.imread(input_path)

    img_h, img_w = image_size(img)

    # ------------------------------------------------------
    # YOLO Prediction
    # ------------------------------------------------------

    results = model.predict(
        source=input_path,
        conf=CONF_THRESHOLD,
        imgsz=IMAGE_SIZE,
        save=True,
        project=RESULT_FOLDER,
        name="predictions",
        exist_ok=True,
        verbose=False,
        max_det=MAX_DETECTIONS
    )

    prediction_dir = os.path.join(
        RESULT_FOLDER,
        "predictions"
    )

    output_path = os.path.join(
        prediction_dir,
        input_filename
    )

    if not os.path.exists(output_path):

        files = [

            f

            for f in os.listdir(prediction_dir)

            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                )
            )

        ]

        if files:

            files.sort(

                key=lambda x: os.path.getmtime(
                    os.path.join(
                        prediction_dir,
                        x
                    )
                ),

                reverse=True

            )

            output_path = os.path.join(
                prediction_dir,
                files[0]
            )

    # ------------------------------------------------------
    # Read Raw Detections
    # ------------------------------------------------------

    raw = []

    if results:

        result = results[0]

        if result.boxes is not None:

            names = result.names

            for box, cls, conf in zip(

                result.boxes.xyxy.tolist(),

                result.boxes.cls.tolist(),

                result.boxes.conf.tolist()

            ):

                if not is_valid_detection(

                    box,

                    conf,

                    img_w,

                    img_h

                ):

                    continue

                label = names[int(cls)].lower()

                raw.append(

                    {

                        "label": label,

                        "confidence": float(conf),

                        "box": box

                    }

                )

    # ------------------------------------------------------
    # Remove Duplicates
    # ------------------------------------------------------

    detections = remove_duplicate_boxes(raw)

    bolt_count = 0

    nut_count = 0

    report_lines = []

    # ------------------------------------------------------
    # Count Objects
    # ------------------------------------------------------

    for det in detections:

        label = det["label"]

        confidence = det["confidence"]

        if label == "bolt":
            bolt_count += 1

        elif label == "nut":
            nut_count += 1

        report_lines.append(
            f"{label.capitalize():<5}   {confidence:.2f}"
        )

    total = bolt_count + nut_count

    # ------------------------------------------------------
    # Build Report
    # ------------------------------------------------------

    if total == 0:

        summary = (
            "==============================\n"
            " AI BOLT & NUT DETECTOR\n"
            "==============================\n\n"
            "No bolts or nuts detected.\n\n"
            "Bolts : 0\n"
            "Nuts  : 0\n"
            "Total : 0"
        )

    else:

        summary = (
            "==============================\n"
            " AI BOLT & NUT DETECTOR\n"
            "==============================\n\n"
            f"Bolts : {bolt_count}\n"
            f"Nuts  : {nut_count}\n"
            f"Total : {total}\n\n"
            "------------------------------\n"
            "Detected Objects\n"
            "------------------------------\n"
            + "\n".join(report_lines)
        )

    # ------------------------------------------------------
    # Load Result Image
    # ------------------------------------------------------

    result_image = None

    if os.path.exists(output_path):

        try:

            result_image = Image.open(output_path).copy()

        except Exception:

            result_image = image

    else:

        result_image = image

    return result_image, summary


# ==========================================================
# FASTAPI
# ==========================================================

app = FastAPI(

    title=PROJECT_NAME,

    version=API_VERSION,

    description="""
AI-powered Bolt & Nut Detection API

Features
--------
• Bolt Detection
• Nut Detection
• Image Preprocessing
• Duplicate Removal
• False Positive Filtering
• FastAPI + Gradio
"""
)

# ==========================================================
# CORS
# ==========================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)


# ==========================================================
# PREDICTION API
# ==========================================================

@app.post("/api/predict")
async def api_predict(
    image: UploadFile = File(...)
):

    ext = os.path.splitext(
        image.filename
    )[1].lower()

    allowed = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    ]

    if ext not in allowed:

        return JSONResponse(
            {
                "success": False,
                "error": "Unsupported image format."
            },
            status_code=400
        )

    unique_id = str(uuid.uuid4())[:8]

    input_filename = f"input_{unique_id}{ext}"

    input_path = os.path.join(
        UPLOAD_FOLDER,
        input_filename
    )

    with open(input_path, "wb") as f:
        shutil.copyfileobj(
            image.file,
            f
        )

    try:

        # ------------------------------------------
        # Image Preprocessing
        # ------------------------------------------

        if ENABLE_PREPROCESSING:
            preprocess_image(input_path)

        img = cv2.imread(input_path)

        img_h, img_w = image_size(img)

        # ------------------------------------------
        # YOLO Prediction
        # ------------------------------------------

        results = model.predict(
            source=input_path,
            conf=CONF_THRESHOLD,
            imgsz=IMAGE_SIZE,
            save=True,
            project=RESULT_FOLDER,
            name="predictions",
            exist_ok=True,
            verbose=False,
            max_det=MAX_DETECTIONS
        )

        prediction_dir = os.path.join(
            RESULT_FOLDER,
            "predictions"
        )

        output_path = os.path.join(
            prediction_dir,
            input_filename
        )

        if not os.path.exists(output_path):

            files = [

                f

                for f in os.listdir(prediction_dir)

                if f.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".webp"
                    )
                )

            ]

            if files:

                files.sort(

                    key=lambda x: os.path.getmtime(
                        os.path.join(
                            prediction_dir,
                            x
                        )
                    ),

                    reverse=True

                )

                output_path = os.path.join(
                    prediction_dir,
                    files[0]
                )

        raw = []

        if results:

            result = results[0]

            if result.boxes is not None:

                names = result.names

                for box, cls, conf in zip(

                    result.boxes.xyxy.tolist(),

                    result.boxes.cls.tolist(),

                    result.boxes.conf.tolist()

                ):

                    if not is_valid_detection(
                        box,
                        conf,
                        img_w,
                        img_h
                    ):
                        continue

                    raw.append(
                        {
                            "label": names[int(cls)].lower(),
                            "confidence": float(conf),
                            "box": box
                        }
                    )

        detections = remove_duplicate_boxes(raw)

        bolt_count = 0
        nut_count = 0

        detection_list = []

        # ------------------------------------------------------
        # Count Objects
        # ------------------------------------------------------

        for det in detections:

            label = det["label"]

            confidence = round(
                det["confidence"],
                2
            )

            if label == "bolt":
                bolt_count += 1

            elif label == "nut":
                nut_count += 1

            detection_list.append(
                {
                    "label": label,
                    "confidence": confidence
                }
            )

        total = bolt_count + nut_count

        # ------------------------------------------------------
        # No Detection
        # ------------------------------------------------------

        if total == 0:

            return JSONResponse(
                {
                    "success": False,
                    "summary": {
                        "bolts": 0,
                        "nuts": 0,
                        "total": 0
                    },
                    "detections": [],
                    "message": "No bolts or nuts detected."
                }
            )

        # ------------------------------------------------------
        # Success Response
        # ------------------------------------------------------

        summary = {

            "bolts": bolt_count,

            "nuts": nut_count,

            "total": total

        }

        result_url = None

        if os.path.exists(output_path):

            result_url = (
                "/api/results/predictions/"
                + os.path.basename(output_path)
            )

        return JSONResponse(

            {

                "success": True,

                "summary": summary,

                "detections": detection_list,

                "result_image_url": result_url,

                "message": "Detection completed successfully."

            }

        )

    except Exception as e:

        return JSONResponse(

            {

                "success": False,

                "error": str(e)

            },

            status_code=500

        )


# ==========================================================
# HEALTH API
# ==========================================================

@app.get("/api/health")

def health():

    return {

        "success": True,

        "status": "running",

        "project": PROJECT_NAME,

        "version": API_VERSION,

        "model": os.path.basename(MODEL_PATH),

        "confidence_threshold": CONF_THRESHOLD,

        "image_size": IMAGE_SIZE,

        "duplicate_removal": ENABLE_DUPLICATE_REMOVAL,

        "preprocessing": ENABLE_PREPROCESSING,

        "supported_classes": SUPPORTED_CLASSES,

        "message": "Bolt & Nut Detector API is running successfully."

    }


# ==========================================================
# RESULT IMAGE API
# ==========================================================

@app.get("/api/results/{subpath:path}")

def serve_results(subpath: str):

    file_path = os.path.join(
        RESULT_FOLDER,
        subpath
    )

    # ------------------------------------------
    # File Exists?
    # ------------------------------------------

    if not os.path.isfile(file_path):

        return JSONResponse(

            {

                "success": False,

                "error": "Result image not found."

            },

            status_code=404

        )

    # ------------------------------------------
    # Supported Media Types
    # ------------------------------------------

    media_types = {

        ".jpg": "image/jpeg",

        ".jpeg": "image/jpeg",

        ".png": "image/png",

        ".bmp": "image/bmp",

        ".webp": "image/webp"

    }

    ext = os.path.splitext(
        file_path
    )[1].lower()

    media_type = media_types.get(

        ext,

        "application/octet-stream"

    )

    # ------------------------------------------
    # Return Image
    # ------------------------------------------

    return FileResponse(

        path=file_path,

        media_type=media_type,

        filename=os.path.basename(file_path)

    )


# ==========================================================
# ROOT API
# ==========================================================

@app.get("/")

def root():

    return {

        "success": True,

        "project": PROJECT_NAME,

        "version": API_VERSION,

        "status": "Running",

        "model": os.path.basename(MODEL_PATH),

        "confidence_threshold": CONF_THRESHOLD,

        "image_size": IMAGE_SIZE,

        "supported_classes": SUPPORTED_CLASSES,

        "features": [

            "Bolt Detection",

            "Nut Detection",

            "Image Preprocessing",

            "Duplicate Removal",

            "False Positive Filtering",

            "FastAPI",

            "Gradio"

        ],

        "endpoints": {

            "Health": "/api/health",

            "Prediction": "/api/predict",

            "Result Image": "/api/results/{filename}",

            "Gradio": "/gradio"

        }

    }


# ==========================================================
# GRADIO USER INTERFACE
# ==========================================================

demo = gr.Interface(

    fn=predict_gradio,

    inputs=gr.Image(

        type="pil",

        label="📤 Upload Bolt & Nut Image"

    ),

    outputs=[

        gr.Image(

            label="🎯 Detection Result",

            type="pil"

        ),

        gr.Textbox(

            label="📊 Detection Report",

            lines=18,

            max_lines=25

        )

    ],

    title="🔩 AI Bolt & Nut Detector",

    description="""
Upload an image containing bolts and/or nuts.

Features
---------
✅ YOLOv8 Detection

✅ Bolt Counting

✅ Nut Counting

✅ Duplicate Removal

✅ False Positive Filtering

✅ Image Preprocessing

✅ Automatic Detection Report

Supported Formats
-----------------
JPG • JPEG • PNG • BMP • WEBP
""",

    examples=[

    ],

    allow_flagging="never",

    cache_examples=False,

    analytics_enabled=False,

    theme=gr.themes.Soft(),

    submit_btn="🚀 Detect",

    clear_btn="🗑 Clear"

)


# ==========================================================
# STARTUP EVENT
# ==========================================================

@app.on_event("startup")

async def startup_event():

    print("\n" + "=" * 65)

    print("🚀 AI BOLT & NUT DETECTOR")

    print("=" * 65)

    print(f"Project          : {PROJECT_NAME}")

    print(f"Version          : {API_VERSION}")

    print(f"Model            : {MODEL_PATH}")

    print(f"Confidence       : {CONF_THRESHOLD}")

    print(f"Image Size       : {IMAGE_SIZE}")

    print(f"Uploads Folder   : {UPLOAD_FOLDER}")

    print(f"Results Folder   : {RESULT_FOLDER}")

    print(f"Duplicate Filter : {ENABLE_DUPLICATE_REMOVAL}")

    print(f"Preprocessing    : {ENABLE_PREPROCESSING}")

    print("=" * 65)

    print("Backend Started Successfully")

    print("=" * 65 + "\n")


# ==========================================================
# SHUTDOWN EVENT
# ==========================================================

@app.on_event("shutdown")

async def shutdown_event():

    print("\n" + "=" * 65)

    print("Stopping AI Bolt & Nut Detector...")

    print("Server Stopped Successfully")

    print("=" * 65 + "\n")


# ==========================================================
# MOUNT GRADIO
# ==========================================================

app = gr.mount_gradio_app(

    app,

    demo,

    path="/gradio"

)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "app:app",

        host="0.0.0.0",

        port=7860,

        reload=False,

        workers=1

    )