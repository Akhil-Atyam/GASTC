import streamlit as st
from streamlit_geolocation import streamlit_geolocation
from ultralytics import YOLO
from PIL import Image
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
import cv2

# Load environment variables
load_dotenv("keys.env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
modelTEXT = genai.GenerativeModel("gemini-2.5-flash-lite")
location = streamlit_geolocation()

# Streamlit page configuration
st.set_page_config(
    page_title="GASTC TEST",
    page_icon="🚮",
    layout="centered"
)
st.title("GASTC test")
image_file = st.camera_input("Take a picture")

# Waste category mapping (what waste goes where)
WASTE_MAP = {
    "PVC pipe": "General Waste",
    "Battery": "Hazardous",
    "cube": "General Waste",
    "fruit": "Compostable",
}

# Category info (disposal location)
CATEGORY_INFO = {
    "Compostable": "Place in a compost bin or local compost facility.",
    "Recyclable": "Place in your recycling bin. Check local rules for plastics, paper, and cardboard.",
    "Hazardous": "Take to a hazardous waste collection site.",
    "General Waste": "Place in the regular trash bin."
}

# Bounding box colors (BGR for OpenCV)
CATEGORY_COLORS = {
    "Compostable": (0, 200, 0),
    "Recyclable": (255, 165, 0),
    "Hazardous": (0, 0, 255),
    "General Waste": (255, 0, 0)
}

# Alias & ignore maps
ALIAS_MAP = {
    "batteries - v1 2023-02-21 10-20pm": "Battery",
    "coral": "PVC pipe"
}

IGNORE_LIST = [
    "Nothing",
    "algae"
]


# YOLO models
models = ["AlgaeCoral.pt",
    "Battery.pt", 
    "Cube.pt", 
    "Fruit.pt", 
    "Ram.pt"
    ]

# Confidence cutoffs per model
MODEL_CUTOFFS = {
    "AlgaeCoral.pt": 0.7,
    "Battery.pt": 0.8,
    "Cube.pt": 0.7,
    "Fruit.pt": 0.6,
    "Ram.pt": 0.5
}


# Detection function
def detect_objects(image_array, model_path):
    model = YOLO(model_path)
    results = model(image_array)
    detections = []

    cutoff = MODEL_CUTOFFS.get(model_path, 0.4)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            name = model.names[class_id]

            name = ALIAS_MAP.get(name, name)
            if name in IGNORE_LIST or confidence < cutoff:
                continue

            category = WASTE_MAP.get(name, "General Waste")
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            detections.append({
                "object": name,
                "confidence": confidence,
                "category": category,
                "position": [x1, y1, x2, y2]
            })

    return detections

# Main logic
if image_file:
    img = Image.open(image_file)
    img_array = np.array(img)

    detections = []
    for model_path in models:
        try:
            detections.extend(detect_objects(img_array, model_path))
        except Exception as e:
            st.error(f"Error loading {model_path}: {e}")

    #Result image with boxes to show where stuff is

    # Convert RGB -> BGR for OpenCV
    draw_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    for det in detections:
        x1, y1, x2, y2 = map(int, det["position"])
        label = f"{det['object']}: {int(det['confidence'] * 100)}% confidence, {det['category']}"
        color = CATEGORY_COLORS.get(det["category"], (255, 255, 255))

        # Box
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 2)

        # Measure text
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Clamp X so text stays on screen
        text_x = max(0, min(x1, draw_img.shape[1] - w - 4))

        # Prefer above box, else put below
        text_y = y1 - 6
        if text_y - h < 0:
            text_y = y2 + h + 6

        # Label background
        cv2.rectangle(
        draw_img,
        (text_x, text_y - h - 4),
        (text_x + w + 4, text_y),
        color,
        -1
        )

        # Label text
        cv2.putText(
        draw_img,
        label,
        (text_x + 2, text_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
        )


    # Convert BGR -> RGB for Streamlit
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    st.image(draw_img, caption="Processed Image with Detections", width=700)

    #Buttons to show disposal location stuff.

    if detections:
        st.subheader("Detected Objects")
        for i, d in enumerate(detections, 1):
            st.markdown(
                f"**{i}. {d['object']}** ({d['confidence']*100:.1f}%) → **{d['category']}**"
            )

            if st.button(f"How to dispose of {d['object']}", key=f"{i}_{d['object']}"):
                st.info(CATEGORY_INFO.get(d["category"], "No info available."))
                response = modelTEXT.generate_content(
                    f"List 1 local center in {location} where I can dispose of {d['category']} waste. "
                    "Only return name and address separated by colon. "
                    "If no location, assume Columbus GA and prefix 'Dispose at:'."
                )
                st.info(response.text)
    else:
        st.warning("No objects detected. Try taking another picture.")
