import logging
import tempfile

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pipeline.preprocess import deskew
from pipeline.detect import detect_rooms
from pipeline.polygons import compute_scale, simplify_polygon, normalize_to_meters, normalize_doors_to_meters
from pipeline.export import build_floorplan_json

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

app = FastAPI(title="Floorplan to 3D")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload_floorplan(
    file: UploadFile = File(...),
):
    """Accept a floor plan image and return a JSON floorplan schema."""
    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": f"Expected image file, got {file.content_type}"},
        )

    # Read with size limit
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024*1024)} MB"},
        )

    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Could not decode image"},
        )

    try:
        # Step 1: Preprocess - deskew
        processed = deskew(image)
        ph, pw = processed.shape[:2]

        # Step 2: Detect rooms and doors
        detection = detect_rooms(processed)
        rooms = detection["rooms"]
        doors = detection["doors"]

        # Step 3: Simplify room polygons
        for room in rooms:
            room["polygon"] = simplify_polygon(room["polygon"])

        # Step 4: Convert to meters (shared scale for consistency)
        scale = compute_scale(pw, ph)
        rooms_m = normalize_to_meters(rooms, pw, ph, scale_m_per_px=scale)
        doors_m = normalize_doors_to_meters(doors, pw, ph, scale_m_per_px=scale)

        # Step 5: Build JSON (include full image dimensions for 2D editor alignment)
        image_width_m = pw * scale
        image_height_m = ph * scale
        result = build_floorplan_json(rooms_m, doors_m, image_width_m, image_height_m)
        return result

    except Exception:
        logger.exception("Pipeline error")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to process floor plan. Try a cleaner image."},
        )


@app.post("/upload-vlm")
async def upload_floorplan_vlm(
    file: UploadFile = File(...),
    model: str = Query("gemini-2.5-flash", description="Gemini model name"),
):
    """Hybrid pipeline: OpenCV geometry + Gemini room labeling.

    Uses OpenCV for accurate room/door detection, then Gemini to
    label each room with a descriptive name based on the image.
    Returns the same schema as /upload (floorplan wrapper).
    """
    if file.content_type and not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": f"Expected image file, got {file.content_type}"},
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024*1024)} MB"},
        )

    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Could not decode image"},
        )

    try:
        from pipeline.vlm_vectorize import label_rooms_with_vlm
        from pipeline.detect import detect_rooms_cv, detect_doors_cv

        # Step 1: OpenCV detection (same as /upload pipeline)
        processed = deskew(image)
        ph, pw = processed.shape[:2]

        cv_rooms = detect_rooms_cv(processed, classify=False)
        cv_doors = detect_doors_cv(processed)

        if not cv_rooms:
            return JSONResponse(
                status_code=500,
                content={"error": "Could not detect rooms. Try a cleaner image."},
            )

        # Step 2: Gemini labels the rooms using the image
        image_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        labeled_rooms = label_rooms_with_vlm(image_rgb, cv_rooms, model_name=model)

        # Step 3: Simplify polygons and convert to meters (same as /upload)
        for room in labeled_rooms:
            room["polygon"] = simplify_polygon(room["polygon"])

        scale = compute_scale(pw, ph)
        rooms_m = normalize_to_meters(labeled_rooms, pw, ph, scale_m_per_px=scale)
        doors_m = normalize_doors_to_meters(cv_doors or [], pw, ph, scale_m_per_px=scale)

        image_width_m = pw * scale
        image_height_m = ph * scale
        result = build_floorplan_json(rooms_m, doors_m, image_width_m, image_height_m)
        return result

    except RuntimeError as e:
        logger.exception("VLM dependency error")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    except Exception:
        logger.exception("VLM pipeline error")
        return JSONResponse(
            status_code=500,
            content={"error": "VLM processing failed. Try a cleaner image."},
        )


@app.get("/health")
async def health():
    return {"status": "ok"}
