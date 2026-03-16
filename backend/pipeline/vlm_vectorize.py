"""
VLM-based floor plan vectorization using Qwen2.5-VL.

Produces a wall-first JSON schema where:
- Walls are defined as independent primitives with unique IDs
- Rooms reference wall IDs (dependency-ordered)
- Openings (doors/windows) reference parent walls with position offsets
"""

import json
import logging
import re

from PIL import Image

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an architectural floor plan vectorization assistant.
Your task is to analyze a raster floor plan image and output a structured JSON
representing the geometric topology.

Rules:
1. Define all walls FIRST as independent primitives with unique IDs (w1, w2, ...).
2. Define rooms SECOND, referencing wall IDs only.
3. Define openings (doors/windows) LAST, each referencing a parent wall.
4. Coordinates are in pixels from the top-left corner of the image.
5. Wall thickness is estimated from the image (typically 10-30px).
6. Output ONLY valid JSON. No explanation, no markdown, no extra text."""

USER_PROMPT = """Vectorize this floor plan image.

Output the following JSON structure exactly:
{
  "scale": { "pixels_per_meter": <estimated> },
  "walls": [
    { "id": "w1", "start": [x, y], "end": [x, y], "thickness": <px> }
  ],
  "rooms": [
    {
      "id": "r1",
      "label": "<room type>",
      "wall_ids": ["w1", ...],
      "floor_polygon": [[x,y], ...]
    }
  ],
  "openings": [
    { "id": "o1", "type": "door|window", "wall_id": "w1", "position": <px from start>, "width": <px> }
  ]
}"""

# Lazy-loaded model and processor
_model = None
_processor = None


def _load_model(model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    """Load the Qwen2.5-VL model and processor (lazy, cached)."""
    global _model, _processor

    if _model is not None and _processor is not None:
        return _model, _processor

    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as e:
        raise RuntimeError(
            "VLM dependencies not installed. Run: "
            "pip install transformers torch qwen-vl-utils"
        ) from e

    logger.info("Loading VLM model: %s", model_name)
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    _processor = AutoProcessor.from_pretrained(model_name)
    logger.info("VLM model loaded successfully")
    return _model, _processor


def vectorize_floorplan(image_path: str, model_name: str | None = None) -> dict | None:
    """
    Vectorize a floor plan image using Qwen2.5-VL.

    Args:
        image_path: Path to the floor plan image.
        model_name: Optional model override (default: Qwen2.5-VL-7B-Instruct).

    Returns:
        Parsed JSON dict with walls, rooms, and openings, or None on failure.
    """
    import torch

    model, processor = _load_model(model_name or "Qwen/Qwen2.5-VL-7B-Instruct")

    image = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.1,
            do_sample=True,
        )

    raw = processor.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    return parse_vlm_output(raw)


def vectorize_floorplan_from_array(
    image_rgb,
    model_name: str | None = None,
) -> dict | None:
    """
    Vectorize from a numpy array (RGB) — used by the upload endpoint.

    Args:
        image_rgb: numpy array in RGB format (H, W, 3).
        model_name: Optional model override.

    Returns:
        Parsed JSON dict or None.
    """
    import torch

    model, processor = _load_model(model_name or "Qwen/Qwen2.5-VL-7B-Instruct")

    image = Image.fromarray(image_rgb)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.1,
            do_sample=True,
        )

    raw = processor.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    return parse_vlm_output(raw)


def parse_vlm_output(raw: str) -> dict | None:
    """Parse and validate VLM text output into the wall-first schema."""
    # Strip markdown fences if model wraps output
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error("JSON parse failed: %s", e)
        logger.debug("Raw VLM output: %s", raw)
        return None

    # Validate required top-level keys
    if not isinstance(data, dict):
        logger.error("VLM output is not a JSON object")
        return None

    if "walls" not in data or "rooms" not in data:
        logger.error("VLM output missing 'walls' or 'rooms'")
        return None

    # Validate and normalize walls
    wall_ids = set()
    for wall in data["walls"]:
        if "id" not in wall or "start" not in wall or "end" not in wall:
            logger.warning("Skipping wall with missing fields: %s", wall)
            continue
        wall_ids.add(wall["id"])
        # Ensure start/end are lists of 2 numbers
        wall["start"] = [float(wall["start"][0]), float(wall["start"][1])]
        wall["end"] = [float(wall["end"][0]), float(wall["end"][1])]
        wall["thickness"] = float(wall.get("thickness", 20))

    # Validate rooms
    for room in data["rooms"]:
        if "id" not in room:
            logger.warning("Room missing 'id': %s", room)
        # Ensure floor_polygon is a list of [x,y] pairs
        if "floor_polygon" in room:
            room["floor_polygon"] = [
                [float(p[0]), float(p[1])] for p in room["floor_polygon"]
            ]

    # Ensure openings exist (may be empty)
    data.setdefault("openings", [])
    data.setdefault("scale", {"pixels_per_meter": 50})

    return data
