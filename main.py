import streamlit as st
from streamlit_geolocation import streamlit_geolocation
from ultralytics import YOLO
from PIL import Image
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
#run command below
#python -m streamlit run main.py
#Page configs.
# Load environment variables from keys.env (keep this file out of version control)
load_dotenv("keys.env")

# Configure the Generative AI client using the GEMINI_API_KEY from the env
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
modelTEXT = genai.GenerativeModel("gemini-2.5-flash-lite")
location = streamlit_geolocation()


st.set_page_config(
    page_title="GASTC TEST",
    page_icon="🚮",
    layout="centered"
)
st.title('GASTC test')
image_file = st.camera_input("Take a picture")

#waste configs
WASTE_MAP = {
    "algae": "Compost",
    "coral": "General",
}
LABEL_ALIAS = { #this is to make some labels more readable considering roboflow names
    "batteries - v1 2023-02-21 10-20pm": "battery"
}
CATEGORY_INFO = {
    "Compost": "Place in a compost bin or local compost facility.",
    "Recycle": "Place in your recycling bin. Check local rules for plastics, paper, and cardboard.",
    "Hazard": "Take to a hazardous waste collection site.",
    "General": "Place in the regular trash bin."
}
CATEGORY_LOCATIONS = {
    "Compost": "Columbus Recycling & Sustainability Center 8001 Pine Grove Way, Columbus, GA 31909",
    "Recycle": "Recycling Center",
    "Hazard": "Hazardous Waste Facility",
    "General": "Trash Bin"
}


model = YOLO("best.pt")
#what our detections run off
def detect_objects(image_path):
    results = model(image_path)
    detections = []
    for result in results:
        # Loop through each detected bounding box
        for box in result.boxes:
            class_id = int(box.cls[0])        # Class ID is cls btw
            confidence = float(box.conf[0])   
            object_name = model.names[class_id]
            category = WASTE_MAP.get(object_name, "General") # use the dictionary to get the value of the key aka our object
            if confidence >= 0.4:  
                detections.append({
                    "object": object_name,
                    "confidence": round(confidence, 2),
                    "category": category
                })
    return detections
#detect_objects("webcam_photo.jpg") ignore this is for testing
if image_file:
    img = Image.open(image_file)

    # To convert PIL Image to numpy array:
    img_array = np.array(img)
   
    detections = detect_objects(img_array)
    #st.write(detections)
    
    if detections:
        st.subheader("Detected Objects:")
        for idx, item in enumerate(detections, start=1):
            obj = item["object"]
            conf = item["confidence"]
            category = item["category"]

            st.markdown(f"**{idx}. {obj}** ({conf*100:.1f}%) -> **{category}**")
            if st.button(f"How to dispose of {obj}", key=f"{idx}_{obj}"):
                st.info(CATEGORY_INFO.get(category, "No info available."))
                response = modelTEXT.generate_content("List 1 local center in " + str(location) + " where I can dispose of " + category + " waste. Do not provide with any extra information, only provide a name of locatoon and an address seperated by a colon .if you cannot find a location, let us know. Do it so every result is in a different line seperate with enter key do not number. If you are not given a location in the sense you are given none for coordinates, say no location found, please click the arrow pointer to enable locations services, do not provide fake details in this case, if the location is weird, assume location is columbus ga")
                st.info(response.text)
    else:
        st.warning("No objects detected. Try taking another picture.")