import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torchvision import transforms
import timm
import streamlit as st
st.set_page_config(page_title="Nail Disease Detection", layout="centered")


# --- Session state pour garder l'onglet actif ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📷 Predict"

# --- Sidebar filters ---
st.sidebar.header("Filtres")
example_classes = [
    "Acral_Lentiginous_Melanoma",
    "Healthy_Nail",
    "Onychogryphosis",
    "blue_finger",
    "clubbing",
    "pitting"
]
selected_classes = st.sidebar.multiselect(
    "Sélectionner les classes à afficher",
    options=example_classes,
    default=example_classes
)

# --- Barre de navigation (onglets persistants) ---
tab_choice = st.radio(
    "Navigation",
    ["📷 Predict", "📊 EDA", "ℹ️ About"],
    index=["📷 Predict", "📊 EDA", "ℹ️ About"].index(
        st.session_state.active_tab),
    horizontal=True
)
st.session_state.active_tab = tab_choice

# --- Modèle ---


@st.cache_resource
def load_model():
    model = timm.create_model('resnet18d', pretrained=False, num_classes=6)
    model.load_state_dict(torch.load('models/model.pth',
                                     map_location=torch.device('cpu')))
    model.eval()
    return model


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def predict(image, model, class_names):
    with torch.no_grad():
        inputs = preprocess_image(image)
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1).numpy().flatten()
        pred_class = np.argmax(probs)
    return class_names[pred_class], probs[pred_class], probs


# --- UI Header ---
st.title("🩺 Nail Disease Detection")
st.markdown(
    "Upload or capture an image of a nail and let AI detect the condition.")

# --- Chargement modèle ---
model = load_model()

# --- Onglet Prédiction ---
if tab_choice == "📷 Predict":
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        label, confidence, all_probs = predict(image, model, example_classes)
        st.success(f"**Prediction:** {label} ({confidence*100:.2f}%)")

        st.subheader("Confidence per class")
        for name, prob in zip(example_classes, all_probs):
            if name in selected_classes:
                st.progress(float(prob), text=f"{name}: {prob*100:.2f}%")

# --- Onglet EDA ---
elif tab_choice == "📊 EDA":
    st.header("📊 Analyse exploratoire des données (EDA)")

    class_counts_dict = {
        "Acral_Lentiginous_Melanoma": 120,
        "Healthy_Nail": 400,
        "Onychogryphosis": 90,
        "blue_finger": 60,
        "clubbing": 100,
        "pitting": 230
    }
    filtered_counts = {cls: class_counts_dict[cls]
                       for cls in selected_classes if cls in class_counts_dict}

    st.subheader("Répartition des classes")
    st.bar_chart(data=filtered_counts)

    st.subheader("Répartition proportionnelle")
    fig1, ax1 = plt.subplots()
    ax1.pie(filtered_counts.values(), labels=filtered_counts.keys(),
            autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    st.subheader("Exemples d'images")
    for cls in selected_classes:
        st.markdown(f"**{cls}**")
        image_folder = f"data/train/{cls}"
        if os.path.exists(image_folder):
            image_files = [f for f in os.listdir(image_folder)
                           if f.endswith((".jpg", ".jpeg", ".png"))]
            image_files = random.sample(image_files, min(3, len(image_files)))
            cols = st.columns(len(image_files))
            for i, img_file in enumerate(image_files):
                img_path = os.path.join(image_folder, img_file)
                image = Image.open(img_path)
                cols[i].image(image, use_column_width=True)
        else:
            st.warning(f"Aucun dossier trouvé pour la classe '{cls}'")

    st.subheader("Distribution simulée des tailles d’images")
    widths = np.random.normal(loc=192, scale=10, size=300)
    heights = np.random.normal(loc=192, scale=10, size=300)
    fig2, ax2 = plt.subplots()
    sns.histplot(widths, color="skyblue", label="Largeurs", kde=True, ax=ax2)
    sns.histplot(heights, color="salmon", label="Hauteurs", kde=True, ax=ax2)
    ax2.legend()
    st.pyplot(fig2)

# --- Onglet À propos ---
elif tab_choice == "ℹ️ About":
    st.write("""
    ### À propos
    Cette application a été développée pour le hackathon "AI for Impact" afin de détecter les maladies des ongles par IA.

    **Fonctionnalités :**
    - Classification d'images médicales
    - Visualisation des probabilités
    - Analyse des données d'entraînement

    **Technologies :**
    - Framework : Streamlit
    - Modèle : ResNet-18
    - Traitement d'image : PyTorch

    **Développé par Junie Claude Bella**
    """)
