# NEW LIME EXPLAINER (explain.py)

import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.applications.efficientnet import preprocess_input


def lime_explanation(model, img_pil):
    """
    Run LIME explanation on a PIL image with EfficientNet preprocessing.
    Returns image with boundaries marked.
    """
    # Resize and preprocess the image
    img_resized = img_pil.resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # LIME instance
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image=img_array[0],
        classifier_fn=model.predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    return mark_boundaries(temp / 255.0, mask)
