import streamlit as st
from streamlit_geolocation import streamlit_geolocation
from ultralytics import YOLOE
from PIL import Image
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
import cv2

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------
st.set_page_config(
    page_title="SmartSort",
    page_icon="🚮",
    layout="centered"
)

# --------------------------------------------------
# ENVIRONMENT VARIABLES + APIS
# --------------------------------------------------
load_dotenv("keys.env")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
modelTEXT = genai.GenerativeModel("gemini-2.5-flash-lite")

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "image" not in st.session_state:
    st.session_state.image = None

if "detections" not in st.session_state:
    st.session_state.detections = []

if "active_info" not in st.session_state:
    st.session_state.active_info = None

# --------------------------------------------------
# WASTE CLASSIFICATION MAP
# --------------------------------------------------
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
    "cup": "Recyclable",
    "fork": "General Waste",
    "spoon": "General Waste",
    "knife": "General Waste",
    "laptop": "Hazardous",
    "remote control": "Hazardous",
    "keyboard": "Hazardous",
    "mouse": "Hazardous",
    "bottle": "Recyclable",
    "glasses": "General Waste",
    "wood block": "General Waste",
    "ball": "General Waste"
}

CATEGORY_INFO = {
    "Compostable": "Place in a compost bin or local compost facility.",
    "Recyclable": "Place in your recycling bin. Check local rules for plastics, paper, and cardboard.",
    "Hazardous": "Take to a hazardous waste collection site.",
    "General Waste": "Place in the regular trash bin."
}

CATEGORY_COLORS = {
    "Compostable": (0, 128, 129),
    "Recyclable": (0, 50, 4),
    "Hazardous": (0, 0, 255),
    "General Waste": (255, 0, 0)
}
# --------------------------------------------------
# YOLOE MODEL
# --------------------------------------------------
yoloe_model = YOLOE("yoloe-26l-seg.pt")

YOLOE_CLASSES = list(WASTE_MAP.keys())

yoloe_model.set_classes(
    YOLOE_CLASSES,
    yoloe_model.get_text_pe(YOLOE_CLASSES)
)

# --------------------------------------------------
# DRAW BOUNDING BOXES
# --------------------------------------------------
def draw_boxes(image, detections):

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for d in detections:

        x1, y1, x2, y2 = d["box"]
        label = f"{d['object']} {d['confidence']*100:.1f}%"
        color = CATEGORY_COLORS.get(d["category"], (200,200,200))

        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)

        (tw,th),_ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )

        ty = y1-8 if y1-th-8 > 0 else y2+th+8

        cv2.rectangle(img,(x1,ty-th-4),(x1+tw+6,ty),color,-1)

        cv2.putText(
            img,
            label,
            (x1+3,ty-4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,255),
            1
        )

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
def go(page):
    st.session_state.page = page

def set_info(obj, cat):
    st.session_state.active_info = (obj, cat)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if st.session_state.page == "home":

    st.title("🚮 SmartSort")
    st.header("Take a photo. We tell you where it goes.", text_alignment="center")

    st.image("trash.png", use_container_width=True)

    st.button(
        "Proceed",
        use_container_width=True,
        on_click=go,
        args=("camera",)
    )

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
            detections = []

            try:
                results = yoloe_model.predict(arr)

                for r in results:

                    for box in r.boxes:

                        conf = float(box.conf[0])
                        cls = int(box.cls[0])

                        raw_name = yoloe_model.names[cls].lower().strip()
                        name = raw_name

                        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()

                        detections.append({
                            "object": name,
                            "confidence": conf,
                            "category": WASTE_MAP.get(name,"General Waste"),
                            "box": [int(x1),int(y1),int(x2),int(y2)]
                        })

            except Exception as e:
                st.warning(f"YOLOE error: {e}")

            st.session_state.detections = detections
            st.session_state.page = "results"

        st.button(
            "Continue",
            use_container_width=True,
            on_click=process
        )

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

    grouped = {}

    for d in st.session_state.detections:
        grouped.setdefault(d["object"], []).append(d)

    for obj,items in grouped.items():

        cat = items[0]["category"]

        st.markdown(f"**{len(items)} {obj} → {cat}**")

        st.button(
            f"How to dispose of {obj}?",
            key=f"dispose_{obj}",
            on_click=set_info,
            args=(obj,cat)
        )

    if st.session_state.detections == []:

        st.warning(
            "No objects detected. Try taking another photo with better lighting or a clearer view of the item."
        )

    if st.session_state.active_info:

        obj,cat = st.session_state.active_info

        st.info(CATEGORY_INFO.get(cat))

        response = modelTEXT.generate_content(
            f"List 1 local center in Columbus, GA where I can dispose of {cat} waste. "
            "Only return name and address separated by colon. prefix Dispose at:"
        )

        st.info(response.text)

    st.button(
        "Take another photo",
        use_container_width=True,
        on_click=go,
        args=("camera",)
    )

    st.button(
        "Return to home",
        use_container_width=True,
        on_click=go,
        args=("home",)
    )