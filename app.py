import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
import io

# Load the model (update path if saved model exists)


@st.cache_resource
def load_model():
    model = timm.create_model('resnet18d', pretrained=False, num_classes=6)
    model.load_state_dict(torch.load(
        'models/model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Define preprocessing (same as training)


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function


def predict(image, model, class_names):
    with torch.no_grad():
        inputs = preprocess_image(image)
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1).numpy().flatten()
        pred_class = np.argmax(probs)
    return class_names[pred_class], probs[pred_class], probs


# UI
st.set_page_config(page_title="Nail Disease Detection", layout="centered")
st.title("🩺 Nail Disease Detection")
st.markdown(
    "Upload or capture an image of a nail and let AI detect the condition.")

# Sidebar for options
st.sidebar.header("Options")
example_classes = [
    "Acral_Lentiginous_Melanoma",
    "Healthy_Nail",
    "Onychogryphosis",
    "blue_finger",
    "clubbing",
    "pitting"
]
class_names = st.sidebar.multiselect(
    "Select Class Labels (ordered)", options=example_classes, default=example_classes)

# Load model
model = load_model()

# Tabs
tab1, tab2 = st.tabs(["📷 Predict", "ℹ️ About"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        label, confidence, all_probs = predict(image, model, class_names)
        st.success(f"**Prediction:** {label} ({confidence*100:.2f}%)")

        st.subheader("Confidence per class")
        for name, prob in zip(class_names, all_probs):
            st.progress(float(prob), text=f"{name}: {prob*100:.2f}%")

with tab2:
    st.write("""
        ### About
        This app was developed for the "AI for Impact" Hackathon to detect nail diseases using deep learning.
        
        - Model: ResNet18d (from TIMM)
        - Input: Nail image (192x192)
        - Output: Predicted disease class
        
        **Made with ❤️ by Junie Claude Bella**
    """)
