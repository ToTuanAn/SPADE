import torch
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from glob import glob
import pandas as pd
import torch.nn.functional as F
from models.pix2pix_model import Pix2PixModel
from options.test_options import TestOptions
import torchvision.transforms as transforms
import cv2

st.set_page_config(layout="wide")

opt = TestOptions().parse()
model = Pix2PixModel(opt)
model.eval()

latent_vecs = []

map_transforms = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])

def convert_to_label(canvas_extract):
    cond_list = [
        canvas_extract == (126.0, 104.0, 195.0),
        canvas_extract == (222.0, 235.0, 241.0),
        canvas_extract == (146.0, 124.0, 88.0),
        canvas_extract == (129.0, 128.0, 126.0),
        canvas_extract == (72.0, 47.0, 10.0),
        canvas_extract == (226.0, 188.0, 16.0),
        canvas_extract == (24.0, 16.0, 226.0),
        canvas_extract == (16.0, 207.0, 226.0),
        canvas_extract == (100.0, 220.0, 180.0),
        canvas_extract == (63.0, 60.0, 3.0),
        canvas_extract == (120.0, 239.0, 95.0),
        canvas_extract == (33.0, 146.0, 10.0),
    ]
    choices = [
        1,
        0,
        2,
        7,
        6,
        8,
        4,
        9,
        10,
        3,
        5,
        11,
    ]
    converted = np.select(cond_list, choices, 1)[:, :, 0]
    return converted.astype(np.uint8)

def get_predictions(canvas_extract):
    print("Plain canvas", canvas_extract.shape)
    label = convert_to_label(canvas_extract)
    label = Image.fromarray(label)
    label = map_transforms(label).squeeze(0)
    label = (label*255).long()
    label_ohe = F.one_hot(label,num_classes=12).permute(2,0,1).to(torch.float32).unsqueeze(0)
    # print(label_ohe)
    data = {}
    data['label'] = label_ohe
    data['image'] = None
    data['instance'] = None

    fake_images = []

    fake_image = model(data, mode='inference')
    fake_image = fake_image.detach().cpu().numpy()
    print("Fake: ", fake_image.shape)
    # fake_image = np.transpose(fake_image,(1,2,0))
    fake_image = np.transpose(fake_image, (0, 2, 3, 1))[0]
    fake_image = (fake_image+1)/2
    fake_image = (fake_image*255).astype(np.uint8)
    print(fake_image)
    fake_image = cv2.resize(fake_image, (400,400), interpolation=cv2.INTER_CUBIC)
    fake_images.append(fake_image)

    return fake_images


color_map = {
    "sky": "#7EB8C3",
    "clouds": "#DEEBF1",
    "mountain": "#927C58",
    "hill": "#81807E",
    "rock": "#482F0A",
    "sand": "#E2BC10",
    "sea": "#1810E2",
    "river": "#10CFE2",
    "water": "#64DCB4",
    "tree": "#3F3C03",
    "grass": "#78EF5F",
    "bush": "#21920A"
}

# init


drawing_mode = "freedraw"
stroke_width = st.sidebar.slider("Stroke width: ", 1, 80, 3)

bg_color = st.sidebar.color_picker("Background color hex: ", color_map["sky"])
brush = st.sidebar.radio(
    "Select brush",
    ("clouds", "sky", "mountain", "hill", "rock", "sand", "sea", "river", "water", "tree", "grass", "bush"))

stroke_color = st.sidebar.color_picker("Stroke color hex: ", color_map[brush])
# print(bg_color)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
# with c1:
st.write("draw here: ")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=None,
    update_streamlit=realtime_update,
    height=450,
    width=450,
    drawing_mode=drawing_mode,
    point_display_radius=0,
    key="canvas",
)
st.write("model output")
c1, c2 = st.columns((1, 1))

print(canvas_result.image_data[:, :, :3][0])
canvas_extract = canvas_result.image_data[:, :, :3].astype(np.float32)

results = get_predictions(canvas_extract)

# st.write(str(converted[0]))

if canvas_result.image_data is not None:
    c1.image(results[0])
if canvas_result.image_data is not None:
    c2.image(results[3])
# if canvas_result.image_data is not None:
#     c3.image(results[2])
# if canvas_result.image_data is not None:
#     c4.image(results[3])
# if canvas_result.image_data is not None:
#     c5.image(results[4])
if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)