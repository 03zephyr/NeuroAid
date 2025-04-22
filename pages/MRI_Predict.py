import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("model/alzheimer_model.h5")  # or .h5

model = load_trained_model()

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

def predict_alzheimer(model, img_array):
    pred = model.predict(img_array)
    class_idx = int(np.argmax(pred, axis=1)[0])
    confidence = float(np.max(pred))
    class_names = ['Demented', 'Non Demented']
    return class_names[class_idx], confidence, class_idx

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(pil_img, heatmap, alpha=0.4):
    img = np.array(pil_img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(superimposed_img)


st.title("NeuroAid 🧠")
st.markdown(f"### Alzheimer's MRI Prediction with Grad-CAM Explanation")
st.write("Upload an MRI image to predict Alzheimer's and see which regions influenced the model's decision.")

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_array, pil_img = preprocess_image(uploaded_file)
    label, confidence, class_idx = predict_alzheimer(model, img_array)
    last_conv_layer_name = 'conv2d_17'  # Change if needed!
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=class_idx)
    gradcam_img = overlay_gradcam(pil_img, heatmap)

    # Show prediction above the images
    st.markdown(f"#### Prediction: **{label}**")
    st.markdown(f"**Confidence:** {confidence:.2%}")

    # Display images side by side, with width control
    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Original MRI", width=256)
    with col2:
        st.image(gradcam_img, caption="Grad-CAM Explanation", width=256)

    st.caption("The highlighted regions most influenced the model's prediction.")
