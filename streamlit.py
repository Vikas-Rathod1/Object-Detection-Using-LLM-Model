




from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import re
from PIL import Image
import cv2
import numpy as np
from collections import Counter
import os


# Streamlit app configuration
st.set_page_config(page_title="Gemini 2.0 Image Analysis App")

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure Google Gemini API
#API_KEY = "AIzaSyBRR80doIuVYWErx9mtFwo1SNRUiocRdsM"
#genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name='gemini-1.5-pro')

def parse_bounding_box(response):
    bounding_boxes = re.findall(r'\[(\d+,\s*\d+,\s*\d+,\s*\d+,\s*[\w\s]+)\]', response)
    parsed_boxes = []
    for box in bounding_boxes:
        parts = box.split(',')
        numbers = list(map(int, parts[:-1]))
        label = parts[-1].strip()
        parsed_boxes.append((numbers, label))
    return parsed_boxes


def draw_bounding_boxes(image, bounding_boxes_with_labels):
    label_colors = {}
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = np.array(image)
    for bounding_box, label in bounding_boxes_with_labels:
        width, height = image.shape[1], image.shape[0]
        ymin, xmin, ymax, xmax = bounding_box
        x1, y1 = int(xmin / 1000 * width), int(ymin / 1000 * height)
        x2, y2 = int(xmax / 1000 * width), int(ymax / 1000 * height)

        if label not in label_colors:
            label_colors[label] = np.random.randint(0, 256, (3,)).tolist()
        color = label_colors[label]

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Position for the label inside the bounding box
        label_bg_x1 = x1
        label_bg_y1 = y1 + 5
        label_bg_x2 = x1 + len(label) * 10  # Adjust width of the label box
        label_bg_y2 = y1 + 20  # Adjust height of the label box

        # Draw a background for the label inside the bounding box
        cv2.rectangle(image, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, -1)

        # Put the label text inside the bounding box
        cv2.putText(image, label, (x1 + 2, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return Image.fromarray(image)

def generate_bounding_box_data(bounding_boxes_with_labels, image_width, image_height):
    data = []
    for bounding_box, label in bounding_boxes_with_labels:
        ymin, xmin, ymax, xmax = bounding_box
        x1, y1 = int(xmin / 1000 * image_width), int(ymin / 1000 * image_height)
        x2, y2 = int(xmax / 1000 * image_width), int(ymax / 1000 * image_height)

        result = {
            "model_version": "1.0",
            "score": "0.85",  # Assuming a confidence score, you can modify this accordingly
            "result": [
                {
                    "id": "1",  # Unique ID for each bounding box
                    "type": "rectangle",
                    "from_name": "object_detection",
                    "to_name": "image",
                    "original_width": str(image_width),
                    "original_height": str(image_height),
                    "image_rotation": "0",
                    "value": {
                        "rotation": "0",
                        "x": str(x1),
                        "y": str(y1),
                        "width": str(x2 - x1),
                        "height": str(y2 - y1),
                        "rectanglelabels": [label]
                    }
                }
            ]
        }
        data.append(result)
    return data

# Streamlit UI
st.title("Object Detection with Google Gemini")
st.write("Upload an image to detect objects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Objects"):
        st.write("Processing...")

        # Send image to Google Gemini API
        response = model.generate_content([
            image,
            '''Return bounding boxes for these objects: Concrete_Block_Wall, Brick_Wall, Vinyl_Siding_Wall, Hung_Window, 
            Casement_Window, Sliding_Window, Bay_Or_Bow_Window, French_Door, Sliding_Glass_Door, Wood_Fence, Vinyl_Fence, 
            Inground_Swimming_Pool, Above_Ground_Swimming_Pool, Electric_Meter, Gas_Meter, Dog, Stairs, House_Number, 
            Single_Door_Garage, Double_Door_Garage, Masonry_Chimney, Pre_Fab_Chimney, Metal_Chimney, Roof_Mounted_Solar_Panel, 
            Ground_Mounted_Solar_Panel, Car, AC_Compressor, Window_AC, Split_AC, Gutter, Shingle_Roof, Metal_Roof, HVAC, 
            Generator, Fuel_Tank, Driveway, Walkway, Metal_Shed, Wooden_Shed, Barn, Sink_Pipe, Copper_Flexible_Hose, 
            Pvc_Flexible_Hose, Pex_Flexible_Hose, Shut_Off_Valve, Floor_Mounted_Toilet, Wall_Hung_Toilet, Shower_Head, Bath_Tub, 
            Above_Ground_Hot_Tub, In_Ground_Hot_Tub, Floor_Mounted_Toilet_Seat, Wall_Hung_Toilet_Seat, Garbage_Disposal, 
            Fire_Extinguisher, Electrical_Panel_Closed, Inside_Electrical_Panel_Cover, Electrical_Breaker_Panel, Fuse_Box_Panel, 
            Thermostat, Furnace, Access_Door_Attic, Emergency_Shutoff_Switch_Hvac, Washer, Dryer, Tankless_Water_Heater, 
            Electric_Water_Heater, Thermal_Expansion_Tank, Pvc_Pipe_Water_Heater, Copper_Pipe_Water_Heater, Water_Detection_Device, 
            Security_Camera, Alarm_System, Smart_Lock, Fire_Alarm in the image in the format: 
            [ymin, xmin, ymax, xmax, object_name]'''
        ])

        result = response.text
        bounding_boxes = parse_bounding_box(result)
        output_image = draw_bounding_boxes(image, bounding_boxes)

        # Count the objects detected
        object_labels = [label for _, label in bounding_boxes]
        object_counts = Counter(object_labels)

        # Generate bounding box data in the required format
        image_width, image_height = image.size
        bounding_box_data = generate_bounding_box_data(bounding_boxes, image_width, image_height)

        # Display the processed image
        st.image(output_image, caption="Processed Image", use_column_width=True)

        # Display the object count below the image
        st.write("### Object Counts:")
        for obj, count in object_counts.items():
            st.write(f"{obj}: {count}")

        # Display the bounding box data
        st.write("### Bounding Box Data:")
        st.json(bounding_box_data)

        st.success("Detection complete!")
