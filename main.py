import streamlit as st
from streamlit_geolocation import streamlit_geolocation
from ultralytics import YOLO
from PIL import Image
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

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
st.title('GASTC test')
image_file = st.camera_input("Take a picture")

# Waste category mapping
WASTE_MAP = {
    "coral": "General",
    "Battery": "Hazard",
    "cube": "General",
    "fruit": "Compost",
}
CATEGORY_INFO = {
    "Compost": "Place in a compost bin or local compost facility.",
    "Recycle": "Place in your recycling bin. Check local rules for plastics, paper, and cardboard.",
    "Hazard": "Take to a hazardous waste collection site.",
    "General": "Place in the regular trash bin."
}

# Alias map: rename long/awkward object names
ALIAS_MAP = {
    "batteries - v1 2023-02-21 10-20pm": "Battery",
    "coral": "PVC pipe"
}

# Ignore list: objects to skip completely
IGNORE_LIST = [
    "nothing",
    "algae"
]

# YOLO model paths
models = ["AlgaeCoral.pt", "Battery.pt", "Cube.pt", "Fruit.pt", "Ram.pt"]

# Confidence cutoffs per model
MODEL_CUTOFFS = {
    "AlgaeCoral.pt": 0.7,  
    "Battery.pt": 0.8,     
    "Cube.pt": 0.7,        
    "Fruit.pt": 0.6,       
    "Ram.pt": 0.5          
}

# Detection function
def detect_objects(image_path, model_path):
    model = YOLO(model_path)
    results = model(image_path)
    detections = []

    # Get cutoff for this model (default to 0.4)
    cutoff = MODEL_CUTOFFS.get(model_path, 0.4)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            object_name = model.names[class_id]

            # Apply alias
            object_name = ALIAS_MAP.get(object_name, object_name)

            # Skip ignored objects
            if object_name in IGNORE_LIST:
                continue

            # Only include if confidence >= cutoff
            if confidence >= cutoff:
                category = WASTE_MAP.get(object_name, "General")
                detections.append({
                    "object": object_name,
                    "confidence": round(confidence, 2),
                    "category": category
                })

    return detections

# Main Streamlit logic
if image_file:
    img = Image.open(image_file)
    img_array = np.array(img)

    # Run detection across all models
    detections = []
    for model_path in models:
        try:
            detections.extend(detect_objects(img_array, model_path))
        except Exception as e:
            st.error(f"Error loading {model_path}: {e}")

    if detections:
        st.subheader("Detected Objects:")
        for idx, item in enumerate(detections, start=1):
            obj = item["object"]
            conf = item["confidence"]
            category = item["category"]

            st.markdown(f"**{idx}. {obj}** ({conf*100:.1f}%) -> **{category}**")
            if st.button(f"How to dispose of {obj}", key=f"{idx}_{obj}"):
                st.info(CATEGORY_INFO.get(category, "No info available."))
                response = modelTEXT.generate_content(
                    f"List 1 local center in {location} where I can dispose of {category} waste. "
                    "Do not provide extra info, only name and address separated by colon. "
                    "If no location, assume columbus ga, affix Dispose at: to your result."
                )
                st.info(response.text)
    else:
        st.warning("No objects detected. Try taking another picture.")
