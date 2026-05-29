from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import numpy as np

# =========================
# Load Trained Model
# =========================
model = load_model("leukemia_densenet.h5")

# =========================
# Test Data Generator
# =========================
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    "test",                 # test/all and test/hem
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# =========================
# Display Class Labels
# =========================
print("\nClass Labels:")
print(test_gen.class_indices)

# =========================
# Predict
# =========================
y_pred_prob = model.predict(test_gen)

# Convert probabilities to binary classes
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# True labels
y_true = test_gen.classes

# =========================
# Evaluation Metrics
# =========================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# =========================
# Print Results
# =========================
print("\n==============================")
print(" LEUKEMIA DETECTION RESULTS ")
print("==============================")

print(f"\n✅ Accuracy  : {accuracy * 100:.2f}%")
print(f"✅ Precision : {precision * 100:.2f}%")
print(f"✅ Recall    : {recall * 100:.2f}%")
print(f"✅ F1-Score  : {f1 * 100:.2f}%")

# =========================
# Confusion Matrix
# =========================
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

st.subheader("📊 Classification Report")

metrics = ["Precision", "Recall", "F1-Score"]
scores = [precision_0, recall_0, f1_0]

for metric, score in zip(metrics, scores):

    st.markdown(f"### {metric}")

    st.progress(float(score))

    st.markdown(
        f"<h4 style='color:green;'>{score:.2f}</h4>",
        unsafe_allow_html=True
    )