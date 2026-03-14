import streamlit as st
from ultralytics import YOLO, YOLOE
from PIL import Image
import numpy as np
import os
import cv2
import groq as Groq  # groq AI client
from dotenv import load_dotenv

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------
st.set_page_config(
    page_title="SmartSort",
    page_icon="🚮",
    layout="centered"
)

# --------------------------------------------------
# ENVIRONMENT VARIABLES / API KEYS
# --------------------------------------------------
load_dotenv("keys.env")

# GROQ API Key (hardcoded color-coded display in UI)
GROQ_KEY = os.getenv("GROQ") or "YOUR_HARD_CODED_KEY"
# Initialize Groq client
client = Groq.Client(api_key=GROQ_KEY)

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
# WASTE MAP
# --------------------------------------------------
WASTE_MAP = {
    "banana": "Compostable",
    "apple": "Compostable",
    "orange": "Compostable",
    "banana peel": "Compostable",
    "apple core": "Compostable",
    "plastic bottle": "Recyclable",
    "glass bottle": "Recyclable",
    "water bottle": "Recyclable",
    "bottle": "Recyclable",
    "tin can": "Recyclable",
    "aluminum can": "Recyclable",
    "can": "Recyclable",
    "paper": "Recyclable",
    "cardboard": "Recyclable",
    "battery": "Hazardous",
    "cell phone": "Hazardous",
    "laptop": "Hazardous",
    "mouse": "Hazardous",
    "remote control": "Hazardous",
    "pvc pipe": "General Waste",
    "cube": "General Waste",
    "plastic bag": "General Waste",
    "food container": "General Waste",
    "styrofoam": "General Waste",
    "fork": "General Waste",
    "spoon": "General Waste",
    "knife": "General Waste",
    "glasses": "General Waste",
    "wood block": "General Waste",
    "ball": "General Waste",
    "watch": "Hazardous"
}

CATEGORY_INFO = {
    "Compostable": "Place in a compost bin.",
    "Recyclable": "Place in recycling.",
    "Hazardous": "Take to hazardous waste center.",
    "General Waste": "Place in regular trash."
}

CATEGORY_COLORS = {
    "Compostable": (0,128,139),
    "Recyclable": (0,100,0),
    "Hazardous": (0,0,255),
    "General Waste": (255,0,0)
}

ALIAS_MAP = {
    "bottle-plastic":"plastic bottle",
    "bottle-glass":"glass bottle",
    "glass-wine":"glass bottle",
    "gym bottle":"water bottle",
    "remote":"remote control",
    "cup": "tin can",
    "Rubiks Cube": "cube"
}

IGNORE_LIST = [
    "person","dog","cat","bird","car","bus","truck",
    "chair","table","desk","bed","sky","tree"
]

# --------------------------------------------------
# YOLO MODELS
# --------------------------------------------------
models = [
    "AlgaeCoral.pt",
    "Cube.pt",
    "Fruit.pt",
    "Ram.pt",
    "Bottle.pt",
    "yolo11n.pt",
    "obj365.pt"
]

MODEL_PRIORITY = {
    "AlgaeCoral.pt":1,
    "Cube.pt":2,
    "Fruit.pt":3,
    "Ram.pt":4,
    "Bottle.pt":5,
    "yolo11n.pt":6,
    "obj365.pt":7
}

MODEL_CUTOFFS = {
    "AlgaeCoral.pt":0.8,
    "Cube.pt":0.5,
    "Fruit.pt":0.4,
    "Ram.pt":0.5,
    "Bottle.pt":0.7,
    "yolo11n.pt":0.4,
    "obj365.pt":0.4
}

# --------------------------------------------------
# YOLOE
# --------------------------------------------------
yoloe_model = YOLOE("yoloe-26l-seg.pt")
YOLOE_CLASSES = list(WASTE_MAP.keys())
yoloe_model.set_classes(YOLOE_CLASSES, yoloe_model.get_text_pe(YOLOE_CLASSES))
YOLOE_CONF = 0.25

# --------------------------------------------------
# IOU + NMS
# --------------------------------------------------
def iou(boxA, boxB):
    xA=max(boxA[0],boxB[0])
    yA=max(boxA[1],boxB[1])
    xB=min(boxA[2],boxB[2])
    yB=min(boxA[3],boxB[3])
    inter=max(0,xB-xA)*max(0,yB-yA)
    if inter==0: return 0
    areaA=(boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB=(boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter/(areaA+areaB-inter)

def non_max_suppression(detections,thresh=0.45):
    detections=sorted(detections,key=lambda d:(d["priority"],d["confidence"]),reverse=True)
    kept=[]
    for d in detections:
        if not any(iou(d["box"],k["box"])>thresh for k in kept):
            kept.append(d)
    return kept

# --------------------------------------------------
# YOLOE DETECTION
# --------------------------------------------------
def detect_yoloe(image):
    results=yoloe_model.predict(image,conf=YOLOE_CONF)
    out=[]
    for r in results:
        for box in r.boxes:
            conf=float(box.conf[0])
            cls=int(box.cls[0])
            name=yoloe_model.names[cls].lower().strip()
            if name not in WASTE_MAP:
                continue
            x1,y1,x2,y2=box.xyxy[0].cpu().numpy()
            out.append({"object":name,"confidence":conf,"category":WASTE_MAP[name],
                        "box":[int(x1),int(y1),int(x2),int(y2)],"priority":10})
    return out

# --------------------------------------------------
# STANDARD YOLO DETECTION
# --------------------------------------------------
def detect_objects(image,model_path):
    model=YOLO(model_path)
    cutoff=MODEL_CUTOFFS.get(model_path,0.4)
    priority=MODEL_PRIORITY.get(model_path,0)
    results=model(image)
    out=[]
    for r in results:
        for box in r.boxes:
            conf=float(box.conf[0])
            if conf<cutoff: continue
            cls=int(box.cls[0])
            raw=model.names[cls].lower().strip()
            name=ALIAS_MAP.get(raw,raw)
            if name in IGNORE_LIST: continue
            if name not in WASTE_MAP: continue
            x1,y1,x2,y2=box.xyxy[0].cpu().numpy()
            out.append({"object":name,"confidence":conf,"category":WASTE_MAP[name],
                        "box":[int(x1),int(y1),int(x2),int(y2)],"priority":priority})
    return out

# --------------------------------------------------
# DRAW BOXES
# --------------------------------------------------
def draw_boxes(image,detections):
    img=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    for d in detections:
        x1,y1,x2,y2=d["box"]
        label=f"{d['object']} {d['confidence']*100:.1f}%"
        color=CATEGORY_COLORS.get(d["category"],(200,200,200))
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
        ty=y1-8 if y1-th-8>0 else y2+th+8
        cv2.rectangle(img,(x1,ty-th-4),(x1+tw+6,ty),color,-1)
        cv2.putText(img,label,(x1+3,ty-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
def go(page): st.session_state.page=page
def set_info(obj,cat): st.session_state.active_info=(obj,cat)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if st.session_state.page=="home":
    st.title("🚮 SmartSort")
    st.header("Take a photo. We tell you where it goes.",text_alignment="center")
    st.image("trash.png",use_container_width=True)
    st.button("Proceed",use_container_width=True,on_click=go,args=("camera",))

# --------------------------------------------------
# CAMERA PAGE
# --------------------------------------------------
elif st.session_state.page=="camera":
    st.header("🚮 Smart Sort")
    st.header("Take a photo",text_alignment="center")

    camera_file = st.camera_input("Take a photo")
    upload_file = st.file_uploader("Or upload an image")

    image_file = camera_file if camera_file else upload_file

    if image_file:
        img = Image.open(image_file)
        st.session_state.image = img

        def process():
            arr = np.array(img)
            all_dets=[]
            try: all_dets.extend(detect_yoloe(arr))
            except Exception as e: st.warning(f"YOLOE error: {e}")
            for m in models:
                try: all_dets.extend(detect_objects(arr,m))
                except Exception as e: st.warning(f"{m} error: {e}")
            st.session_state.detections=non_max_suppression(all_dets)
            st.session_state.page="results"

        st.button("Continue",use_container_width=True,on_click=process)

# --------------------------------------------------
# RESULTS PAGE
# --------------------------------------------------
elif st.session_state.page=="results":
    st.header("🚮 Smart Sort")
    st.header("Results",text_alignment="center")

    boxed=draw_boxes(np.array(st.session_state.image),st.session_state.detections)
    st.image(boxed,use_container_width=True)

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


    for obj,items in grouped.items():
        cat=items[0]["category"]
        st.markdown(f"**{len(items)} {obj} → {cat}**")
        st.button(f"How to dispose of {obj}?",key=f"dispose_{obj}",on_click=set_info,args=(obj,cat))

    if st.session_state.detections==[]:
        st.warning("No objects detected.")

    if st.session_state.active_info:
        obj,cat=st.session_state.active_info
        st.info(CATEGORY_INFO.get(cat))
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",  # or another Groq‑supported model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"List 1 place in Columbus, GA to dispose of {cat} waste. Return only name and address."}
                ],
            )

            st.info(response.choices[0].message.content)
        except Exception as e:
            st.warning(f"API error: {e}")

    st.button("Take another photo",use_container_width=True,on_click=go,args=("camera",))
    st.button("Return to home",use_container_width=True,on_click=go,args=("home",))