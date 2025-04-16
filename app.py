
import streamlit as st
from PIL import Image
import torch
import numpy as np
import json

# Load helper functions
from torchvision import models
import torch.nn.functional as F

# Function to process the image
def process_image(image):
    img = image.resize((255, int(255 * image.height / image.width)) if image.width < image.height else (int(255 * image.width / image.height), 255))
    width, height = img.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    img = img.crop((left, top, right, bottom))
    img = np.array(img).transpose((2, 0, 1)) / 255.0
    img[0] = (img[0] - 0.485) / 0.229
    img[1] = (img[1] - 0.456) / 0.224
    img[2] = (img[2] - 0.406) / 0.225
    return torch.tensor(img).float()
import os
import gdown
import streamlit as st

# Google Drive file ID for checkpoint.pth
file_id = "1saG8eoD9CXk8mHFfhl4p5AgM69veeBbl"
url = f"https://drive.google.com/uc?id={file_id}"
output = "checkpoint.pth"

# Download the model checkpoint if it doesn't exist
if not os.path.exists(output):
    st.text("Downloading model... please wait.")
    gdown.download(url, output, quiet=False)




# Load model from checkpoint
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_mappings']
    model.eval()
    return model

# Predict function
def predict(image, model, cat_to_name, topk=5):
    img_tensor = process_image(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probs, indices = torch.topk(F.softmax(output, dim=1), topk)
    probs = probs.squeeze().tolist()
    indices = indices.squeeze().tolist()
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in indices]
    names = [cat_to_name.get(c, c) for c in classes]
    return probs, names

# Streamlit UI
st.title("🌸 Flower Image Classifier")

st.markdown("This AI-powered app identifies flower species from images using a deep learning model trained on 102 flower categories.")


# Load model and class names
model = load_model("checkpoint.pth")
with open("cat_to_name.json", "r") as f:
    cat_to_name = json.load(f)

uploaded_file = st.file_uploader("Upload an image of a flower", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    
    probs, names = predict(image, model, cat_to_name)
    
    st.markdown("### Top 5 Predictions:")
    for prob, name in zip(probs, names):
        st.write(f"{name}: {prob * 100:.2f}%")
