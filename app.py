import streamlit as st
import torch
from PIL import Image

from models.cnn import VGG11
from models.resnet import get_resnet18
from inference.predict import predict_topk
from utils.transforms import get_transform
from utils.dog_check import is_dog

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

@st.cache_resource
def load_models(num_classes):
    cnn = VGG11(num_classes=num_classes)
    cnn.load_state_dict(torch.load("checkpoints/cnn_model.pth", map_location=device))
    cnn.eval().to(device)

    resnet = get_resnet18(num_classes=num_classes)
    resnet.load_state_dict(torch.load("checkpoints/resnet18_model.pth", map_location=device))
    resnet.eval().to(device)

    return cnn, resnet

with open("class_names.txt") as f:
    class_names = [line.strip() for line in f]

cnn, resnet = load_models(len(class_names))

# ------------------
# UI
# ------------------
st.title("🐶 What's That Dawg?")
st.write("Upload an image to identify dog breeds or check if it’s a dog.")

model_choice = st.selectbox("Choose model", ["CNN", "ResNet-18"])
uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=300)

    # Model + transform
    if model_choice == "CNN":
        model = cnn
        transform = get_transform(128)
    else:
        model = resnet
        transform = get_transform(224)

    img_tensor = transform(image).unsqueeze(0).to(device)

    if st.button("Run Inference"):
        with torch.no_grad():
            outputs = model(img_tensor)

        # ------------------
        # Dog Check
        # ------------------
        is_dog_flag, confidence, pred_idx = is_dog(outputs)

        st.subheader("Dog Check")
        if not is_dog_flag:
            st.error(f"⚠️ This might NOT be a dog (confidence: {confidence:.2f})")
        else:
            st.success(f"🐶 Dog detected (confidence: {confidence:.2f})")

            # ------------------
            # Top-3 Predictions
            # ------------------
            st.subheader("Top-3 Breed Predictions")
            top3 = predict_topk(outputs, class_names, k=3)
            for breed, prob in top3:
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.markdown(f"**{breed}**")
                with col2:
                    st.progress(min(int(prob * 100), 100))
                    st.caption(f"{prob:.2%}")


