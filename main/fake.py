import streamlit as st
import tempfile

try:
    import joblib
except ImportError:
    joblib = None

# -----------------------------
# Helper Function
# -----------------------------
def try_load(path):
    if joblib is None:
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
model = try_load("fake_email_model.pkl")
vectorizer = try_load("tfidf_vectorizer.pkl")

# -----------------------------
# Streamlit Configuration
# -----------------------------
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

if joblib is None:
    st.error("⚠️ The `joblib` library is missing from the environment. Please ensure `joblib` is in `requirements.txt` and reboot your Streamlit app.")

# -----------------------------
# Upload Model if Missing
# -----------------------------
uploaded_model = None
uploaded_vectorizer = None

if model is None or vectorizer is None:

    st.warning("Model or Vectorizer not found. Please upload the required .pkl files below to use the detector.")

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

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tf:
        tf.write(file.getvalue())
        tf.flush()
        return try_load(tf.name)

if uploaded_model:
    model = load_uploaded(uploaded_model)

if uploaded_vectorizer:
    vectorizer = load_uploaded(uploaded_vectorizer)

# -----------------------------
# Email Input
# -----------------------------
email_text = st.text_area(
    "Paste the Email Content",
    height=250,
    placeholder="Paste the complete email here..."
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Detect Email"):

    if email_text.strip() == "":
        st.warning("Please enter an email.")
    else:

        if model is not None and vectorizer is not None:

            try:

                email_vector = vectorizer.transform([email_text])

                prediction = model.predict(email_vector)[0]

                probability = model.predict_proba(email_vector).max() * 100

                if prediction == 1:

                    st.error("🚨 Fake / Phishing Email Detected")

                else:

                    st.success("✅ Legitimate Email")

                st.info(f"Confidence : {probability:.2f}%")

            except Exception as e:

                st.error(f"Prediction Failed: {e}")

        else:

            st.error("Model or Vectorizer not loaded. Please upload the model files above.")
