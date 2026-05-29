import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="Bulk Prediction",
    layout="wide"
)

# ===============================
# LOAD MODEL
# ===============================

model = load_model("leukemia_densenet.h5")

# ===============================
# TITLE
# ===============================

st.markdown("""
<h1 style='text-align:center; color:#C62828;'>
🩸 Bulk Leukemia Prediction Dashboard
</h1>
""", unsafe_allow_html=True)


st.divider()

# ===============================
# FILE UPLOADER
# ===============================

uploaded_files = st.file_uploader(
    "Upload Multiple Images",
    type=["jpg", "png", "jpeg", "bmp"],
    accept_multiple_files=True
)

# ===============================
# PREDICTION SECTION
# ===============================

if uploaded_files:

    predictions = []
    image_names = []

    st.subheader("Prediction Results")

    cols = st.columns(3)

    for idx, uploaded_file in enumerate(uploaded_files):

        # Open image
        img = Image.open(uploaded_file).convert("RGB")

        # Display image
        cols[idx % 3].image(
            img,
            caption=uploaded_file.name,
            use_container_width=True
        )

        # Preprocess image
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array, verbose=0)[0][0]

        predictions.append(float(prediction))
        image_names.append(uploaded_file.name)

    st.divider()

    # ===============================
    # RESULT TABLE
    # ===============================

    result_data = []

    for name, pred in zip(image_names, predictions):

        if pred > 0.5:
            result = "Leukemia Detected"
        else:
            result = "Normal"

        result_data.append({
            "Image": name,
            "Prediction Score": round(pred * 100, 2),
            "Result": result
        })

    df = pd.DataFrame(result_data)

    st.subheader("Prediction Summary Table")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.divider()

    # ===============================
    # LINE CHART
    # ===============================

    st.subheader("Prediction Probability Chart")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        image_names,
        predictions,
        marker='o',
        linewidth=3
    )

    ax.axhline(
        y=0.5,
        linestyle='--'
    )

    ax.set_ylim(0, 1)

    ax.set_xlabel("Images")
    ax.set_ylabel("Prediction Probability")

    ax.set_title("Leukemia Prediction for Uploaded Images")

    plt.xticks(rotation=30)

    st.pyplot(fig)

    st.divider()

    # ===============================
    # BAR CHART
    # ===============================

    st.subheader("Prediction Distribution")

    fig2, ax2 = plt.subplots(figsize=(12, 5))

    ax2.bar(
        image_names,
        predictions
    )

    ax2.set_ylim(0, 1)

    ax2.set_xlabel("Images")
    ax2.set_ylabel("Probability")

    ax2.set_title("Prediction Scores")

    plt.xticks(rotation=30)

    st.pyplot(fig2)

    st.success("Bulk prediction completed successfully.")