# 🩺 Nail Disease Detection with AI

Ce projet a été développé dans le cadre du hackathon **"AI for Impact - Deep Learning & Data Stories"**.  
L’objectif est de fournir une application d’**aide au diagnostic des maladies des ongles** grâce à un modèle de deep learning léger, interprétable, et accessible via une interface web.

## 🚀 Démo en ligne

👉 [Accéder à l’application Streamlit](https://d-tection-maladies-ongles.streamlit.app/)

---

## 🧠 Objectif du projet

Les maladies unguéales sont fréquentes mais souvent négligées ou mal diagnostiquées.  
Cette application permet :

- D’analyser une photo d’ongle
- De prédire l’une des **6 pathologies** (ou un ongle sain)
- D’expliquer visuellement la prédiction avec **LIME**

---

## 📁 Dataset utilisé

- **Nom** : Nail Disease Detection (Kaggle)
- **Classes** :
  - `Healthy_Nail`
  - `Onychogryphosis`
  - `Acral_Lentiginous_Melanoma`
  - `clubbing`
  - `pitting`
  - `blue_finger`
- Dataset déséquilibré → solutions : data augmentation, visualisation

---

## 🧪 Technologies utilisées

| Élément          | Stack                                             |
| ---------------- | ------------------------------------------------- |
| Modèle           | `ResNet18d` via `timm` (PyTorch)                  |
| Prétraitement    | `torchvision.transforms` + Normalisation ImageNet |
| Interprétabilité | `LIME` + `scikit-image`                           |
| Interface        | `Streamlit`                                       |
| Visualisation    | `matplotlib`, `seaborn`, `PIL`, `OpenCV`          |
| Déploiement      | `Streamlit Cloud`                                 |

---

## 🖼 Fonctionnalités principales

- 📁 Upload d’image ou 📸 capture via webcam (`st.camera_input`)
- 🧠 Prédiction instantanée avec score de confiance
- 🧠 Explication visuelle avec LIME (superposition colorée)
- 📊 Onglet EDA avec histogrammes, camemberts, et aperçu d’images
- ℹ️ Onglet "À propos" pour le contexte

---

## 📦 Installation locale

1. Clone le repo :

```bash
git clone https://github.com/Nanji-Freddy/D-tection-Maladies-Ongles-IA.git
cd D-tection-Maladies-Ongles-IA
```
