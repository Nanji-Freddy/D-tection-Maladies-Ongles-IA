# explain.py

import numpy as np
import torch
from lime import lime_image
from PIL import Image
from torchvision import transforms
from skimage.segmentation import mark_boundaries
import streamlit as st


def explain_with_lime(image_pil, model, class_names):
    try:
        image_np = np.array(image_pil)

        # Barre de chargement Streamlit
        with st.spinner("🧠 Génération de l’explication LIME en cours..."):
            def batch_predict(images):
                model.eval()
                transform = transforms.Compose([
                    transforms.Resize((192, 192)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                ])
                batch = torch.stack(
                    [transform(Image.fromarray(img)).float() for img in images], dim=0)
                with torch.no_grad():
                    outputs = model(batch)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                return probs.numpy()

            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                image_np,
                batch_predict,
                top_labels=1,
                hide_color=0,
                num_samples=1000
            )

            top_label = explanation.top_labels[0]
            temp, mask = explanation.get_image_and_mask(
                top_label,
                positive_only=True,
                num_features=10,
                hide_rest=False
            )

        return temp, mask

    except Exception as e:
        st.error(f"❌ Erreur dans LIME : {e}")
        return None, None
