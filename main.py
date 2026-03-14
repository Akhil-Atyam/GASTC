import streamlit as st
from streamlit_geolocation import streamlit_geolocation  # (imported but not currently used)
from ultralytics import YOLO
from PIL import Image
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
import cv2

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------
# Sets browser tab title, icon, and layout
st.set_page_config(
    page_title="SmartSort",
    page_icon="🚮",
    layout="centered"
)

# --------------------------------------------------
# ENVIRONMENT VARIABLES + APIS
# --------------------------------------------------
# Load API keys from keys.env
load_dotenv("keys.env")

# Configure Gemini API for text generation
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
modelTEXT = genai.GenerativeModel("gemini-2.5-flash-lite")

# --------------------------------------------------
# SESSION STATE (constant variables basically)
# --------------------------------------------------
# Tracks which page the user is on
if "page" not in st.session_state:
    st.session_state.page = "home"

# Stores the captured image
if "image" not in st.session_state:
    st.session_state.image = None

# Stores final filtered detections
if "detections" not in st.session_state:
    st.session_state.detections = []

# Stores which object’s disposal info is currently open
if "active_info" not in st.session_state:
    st.session_state.active_info = None

# --------------------------------------------------
# DATA (WASTE CLASSIFICATION MAPS)
# --------------------------------------------------
# Maps detected object → disposal category
WASTE_MAP = {
    "pvc pipe": "General Waste",
    "battery": "Hazardous",
    "cube": "General Waste",
    "banana": "Compostable",
    "plastic bottle": "Recyclable",
    "tin can": "Recyclable",
    "apple": "Compostable",
    "cell phone": "Hazardous",
    "plastic bag": "General Waste",
    "paper": "Recyclable",
    "cardboard": "Recyclable",
    "food container": "General Waste",
    "styrofoam": "General Waste",
    "glass bottle": "Recyclable",
    "wine bottle": "Recyclable",
    "aluminum can": "Recyclable",
    "can": "Recyclable",
    "banana peel": "Compostable",
    "orange": "Compostable",
    "apple core": "Compostable",
    "cup": "General Waste",
    "fork": "General Waste",
    "spoon": "General Waste",
    "knife": "General Waste",
    "laptop": "Hazardous",
    "remote control": "Hazardous",
    "keyboard": "Hazardous",
    "mouse": "Hazardous",
    "bottle": "Recyclable",
    "Tin Can": "Recyclable",
    "cup": "Recyclable"

    
}

# Text explanations for each category
CATEGORY_INFO = {
    "Compostable": "Place in a compost bin or local compost facility.",
    "Recyclable": "Place in your recycling bin. Check local rules for plastics, paper, and cardboard.",
    "Hazardous": "Take to a hazardous waste collection site.",
    "General Waste": "Place in the regular trash bin."
}

# Colors used for bounding boxes (BGR format for OpenCV)
CATEGORY_COLORS = {
    "Compostable": (0, 128, 129),
    "Recyclable": (0, 50, 4),
    "Hazardous": (0, 0, 255),
    "General Waste": (255, 0, 0)
}

# Change weird dataset names to readable ones
ALIAS_MAP = {
    "batteries - v1 2023-02-21 10-20pm": "Battery",
    "coral": "PVC pipe",
    "bottle-glass": "Tin Can",
    "bottle-plastic": "Plastic Bottle",
    "cup-disposable": "Tin Can",
    "cup-handle": "Tin Can",
    "glass-mug": "Tin Can",
    "glass-normal": "Tin Can",
    "glass-wine": "Tin Can",
    "gym bottle": "Tin Can",
    "tin can": "Tin Can",
    "remote": "Cell Phone",
    "non-valuabe waste: batteries": "Battery",
    "canned" : "Tin Can",
    "cup" : "Water Bottle"

}

# Classes that should be ignored entirely
IGNORE_LIST = [
    "nothing",
    "algae",
    "person",
    "dog", "cat", "bird", "horse",
    "car", "bus", "truck", "motorcycle", "bicycle",
    "chair", "sofa", "bed", "table", "dining table", "desk", "cabinet/shelf", "Cabinet/Shelf"
    "tv", "monitor",
    "tree", "plant", "grass",
    "road", "building", "wall", "window", "door",
    "sky", "cloud",
    "backpack", "handbag", "suitcase", 
    "balloon",
    "hat", "book", "sneakers", "other shoes", "lamp", "fan", "bench", "power outlet", "tv"
]


# --------------------------------------------------
# YOLO MODELS
# --------------------------------------------------
models = [
    "AlgaeCoral.pt",
    #"Battery.pt",
    "Cube.pt",
    "Fruit.pt",
    "Ram.pt",
    "Bottle.pt",
    "yolo11n.pt",
    "obj365.pt"
]

# Model priority (higher = better)
MODEL_PRIORITY = {
    #"Battery.pt": 3,
    "Bottle.pt": 5,
    "Fruit.pt": 4,
    "Cube.pt": 0,
    "Ram.pt": 6,
    "AlgaeCoral.pt": 1,
    "yolo11n.pt": 7,
    "obj365.pt": 8
}

# Confidence cutoffs per model
MODEL_CUTOFFS = {
    "AlgaeCoral.pt": 0.8,
    #"Battery.pt": 0.7,
    "Cube.pt": 0.5,
    "Fruit.pt": 0.2,
    "Ram.pt": 0.5,
    "Bottle.pt": 0.7,
    "yolo11n.pt": 0.4,
    "obj365.pt": 0.4
}
#obj to do
# Recycle : Coke can, water bottle
# Compost : Apple, banana done done
# Hazard : Cell phone, battery cell phone done
# General : PVC pipe, cube  cube done pvc done 

# --------------------------------------------------
# IOU + NON-MAX SUPPRESSION
# --------------------------------------------------
# Computes overlap ratio between two bounding boxes
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)

# Removes overlapping boxes, keeping higher-priority detections
def non_max_suppression(detections, thresh=0.45):
    detections = sorted(
        detections,
        key=lambda d: (d["priority"], d["confidence"]),
        reverse=True
    )
    kept = []
    for d in detections:
        if not any(iou(d["box"], k["box"]) > thresh for k in kept):
            kept.append(d)
    return kept

# --------------------------------------------------
# OBJECT DETECTION
# --------------------------------------------------
# Runs YOLO inference for a single model
def detect_objects(image_array, model_path):
    model = YOLO(model_path)
    cutoff = MODEL_CUTOFFS.get(model_path, 0.4)
    priority = MODEL_PRIORITY.get(model_path, 0)
    results = model(image_array)
    out = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < cutoff:
                continue

            cls = int(box.cls[0])

            raw_name = model.names[cls].lower().strip()
            name = ALIAS_MAP.get(raw_name, raw_name)

            if name.lower() in IGNORE_LIST:
                continue


            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            out.append({
                "object": name,
                "confidence": conf,
                "category": WASTE_MAP.get(name, "General Waste"),
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "priority": priority
            })
    return out

# --------------------------------------------------
# DRAW BOUNDING BOXES
# --------------------------------------------------
# Draws boxes and labels on the image
def draw_boxes(image, detections):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = f"{d['object']} {d['confidence']*100:.1f}%"
        color = CATEGORY_COLORS.get(d["category"], (200, 200, 200))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y1 - 8 if y1 - th - 8 > 0 else y2 + th + 8
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty), color, -1)
        cv2.putText(img, label, (x1 + 3, ty - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --------------------------------------------------
# NAVIGATION HELPERS
# --------------------------------------------------
# Changes page safely without double-click issues (session state bug on docs)
def go(page):
    st.session_state.page = page

# Stores which object’s disposal info to show
def set_info(obj, cat):
    st.session_state.active_info = (obj, cat)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if st.session_state.page == "home":
    st.title("🚮 SmartSort")
    st.header("Take a photo. We tell you where it goes.", text_alignment="center")
    st.image("trash.png", use_container_width=True)
    st.button("Proceed", use_container_width=True, on_click=go, args=("camera",))

# --------------------------------------------------
# CAMERA PAGE
# --------------------------------------------------
elif st.session_state.page == "camera":
    st.header("🚮 Smart Sort")
    st.header("Take a photo", text_alignment="center")

    image_file = st.camera_input("")

    if image_file:
        img = Image.open(image_file)
        st.session_state.image = img

        def process():
            arr = np.array(img)
            all_dets = []
            for m in models:
                try:
                    all_dets.extend(detect_objects(arr, m))
                except Exception as e:
                    st.warning(f"{m}: {e}")

            # Apply non-max suppression to remove overlaps
            st.session_state.detections = non_max_suppression(all_dets)
            st.session_state.page = "results"

        st.button("Continue", use_container_width=True, on_click=process)

# --------------------------------------------------
# RESULTS PAGE
# --------------------------------------------------
elif st.session_state.page == "results":
    st.header("🚮 Smart Sort")
    st.header("Results", text_alignment="center")

    boxed = draw_boxes(
        np.array(st.session_state.image),
        st.session_state.detections
    )
    st.image(boxed, use_container_width=True)
    
    counts={
        "Compostable":0,
        "Recyclable":0,
        "Hazardous":0,
        "General Waste":0
    }

    grouped={}
    for d in st.session_state.detections:
        grouped.setdefault(d["object"],[]).append(d)
        counts[d["category"]]+=1
    st.write(f"Item Counts and keys")
    st.write(f"🟩 Recyclable: {counts['Recyclable']}")
    st.write(f"🟨 Compostable: {counts['Compostable']}")
    st.write(f"🟥 Hazardous: {counts['Hazardous']}")
    st.write(f"🟦 General Waste: {counts['General Waste']}")

    for obj, items in grouped.items():
        cat = items[0]["category"]
        st.markdown(f"**{len(items)} {obj} → {cat}**")
        st.button(
            f"How to dispose of {obj}?",
            key=f"dispose_{obj}",
            on_click=set_info,
            args=(obj, cat)
        )

    if st.session_state.detections == []:
        st.warning("No objects detected. Try taking another photo with better lighting or a clearer view of the item.")

    if st.session_state.active_info:
        obj, cat = st.session_state.active_info
        st.info(CATEGORY_INFO.get(cat))
        response = modelTEXT.generate_content(
                    f"List 1 local center in columbus, ga where I can dispose of {d['category']} waste. "
                    "Only return name and address separated by colon. prefix Dispose at:"
                )
        st.info(response.text)


    st.button("Take another photo", use_container_width=True, on_click=go, args=("camera",))
    st.button("Return to home", use_container_width=True, on_click=go, args=("home",))