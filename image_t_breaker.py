import cv2
import torch
import numpy as np
from PIL import Image
from transformers import (
    TableTransformerForObjectDetection,
    DetrImageProcessor
)

# ==============================
# CONFIG
# ==============================
MODEL_NAME = "microsoft/table-transformer-structure-recognition"
MIN_IMAGE_SIZE = 800     # critical for small images
THRESHOLD = 0.5

# ==============================
# UPSCALE FUNCTION (CRITICAL)
# ==============================
def upscale_if_needed(img, min_size=800):
    h, w = img.shape[:2]
    scale = max(min_size / h, min_size / w, 1.0)

    if scale > 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        print(f"[INFO] Upscaled image to {new_w}x{new_h}")

    return img

# ==============================
# LOAD MODEL
# ==============================
processor = DetrImageProcessor.from_pretrained(
    MODEL_NAME,
    size=500,
    max_size=1600
)

model = TableTransformerForObjectDetection.from_pretrained(MODEL_NAME)
model.eval()

# ==============================
# MAIN FUNCTION
# ==============================
def analyze_table(image_path):
    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError("Could not read image")

    # Fix small images
    img_bgr = upscale_if_needed(img_bgr, MIN_IMAGE_SIZE)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)

    # Prepare input
    inputs = processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process
    target_sizes = torch.tensor([pil_image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs,
        threshold=THRESHOLD,
        target_sizes=target_sizes
    )[0]

    print(f"\nDetected {len(results['boxes'])} objects\n")

    # Color map
    COLORS = {
        "table": (255, 255, 0),
        "table row": (255, 0, 0),
        "table column": (0, 255, 0),
        "table column header": (255, 0, 255),
        "table cell": (0, 0, 255)
    }

    elements = []

    for score, label, box in zip(
        results["scores"],
        results["labels"],
        results["boxes"]
    ):
        label_name = model.config.id2label[label.item()]
        conf = round(score.item(), 3)
        box = box.cpu().numpy().astype(int)

        print(f"{label_name:25s} {conf}  {box.tolist()}")

        color = COLORS.get(label_name, (200, 200, 200))

        # Draw box
        cv2.rectangle(
            img_bgr,
            (box[0], box[1]),
            (box[2], box[3]),
            color,
            2
        )

        # Label
        cv2.putText(
            img_bgr,
            f"{label_name} {conf}",
            (box[0], max(box[1] - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        elements.append({
            "label": label_name,
            "confidence": conf,
            "bbox": box.tolist()
        })

    # Save output
    out_path = image_path.replace(".png", "_detected.png")
    cv2.imwrite(out_path, img_bgr)

    print(f"\nSaved result to: {out_path}")

    return elements

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    image_path = "x.png"
    elements = analyze_table(image_path)
