import streamlit as st
import tempfile

# Page Configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Fake Email Detection Bot",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Fake Email Detection Bot")
st.write(
    "Detect whether an email is **Legitimate** or **Phishing/Fake** using Machine Learning."
)

st.divider()

# Safe imports
joblib = None
try:
    import joblib
except Exception as e:
    st.warning(f"Notice: joblib loading info: {e}")

# Helper Function
def try_load(path):
    if joblib is None:
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

# Load Model & Vectorizer
model = try_load("fake_email_model.pkl")
vectorizer = try_load("tfidf_vectorizer.pkl")

# Upload Model if Missing
uploaded_model = None
uploaded_vectorizer = None

if model is None or vectorizer is None:
    st.info("💡 Model files (`fake_email_model.pkl` / `tfidf_vectorizer.pkl`) not found locally. Please upload them below to start testing.")

    col1, col2 = st.columns(2)
    with col1:
        if model is None:
            uploaded_model = st.file_uploader(
                "Upload fake_email_model.pkl",
                type=["pkl"]
            )
    with col2:
        if vectorizer is None:
            uploaded_vectorizer = st.file_uploader(
                "Upload tfidf_vectorizer.pkl",
                type=["pkl"]
            )

def load_uploaded(file):
    if file is None or joblib is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tf:
            tf.write(file.getvalue())
            tf.flush()
            return try_load(tf.name)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

if uploaded_model:
    model = load_uploaded(uploaded_model)

if uploaded_vectorizer:
    vectorizer = load_uploaded(uploaded_vectorizer)

# Email Input
email_text = st.text_area(
    "Paste the Email Content",
    height=200,
    placeholder="Paste the complete email content here..."
)

# Prediction
if st.button("Detect Email"):
    if not email_text.strip():
        st.warning("Please enter email text before analyzing.")
    elif model is None or vectorizer is None:
        st.error("Model or Vectorizer is not loaded yet. Please upload the required `.pkl` model files above.")
    else:
        try:
            email_vector = vectorizer.transform([email_text])
            prediction = model.predict(email_vector)[0]
            
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(email_vector).max() * 100
                conf_str = f" (Confidence: {probability:.2f}%)"
            else:
                conf_str = ""

            if prediction == 1:
                st.error(f"🚨 Fake / Phishing Email Detected{conf_str}")
            else:
                st.success(f"✅ Legitimate Email{conf_str}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
