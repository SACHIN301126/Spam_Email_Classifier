import streamlit as st
import joblib
import time

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="wide"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.title {
    font-size:45px;
    font-weight:bold;
    color:#0E4BF1;
    text-align:center;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
    margin-bottom:25px;
}

.result-box{
    padding:15px;
    border-radius:12px;
    font-size:22px;
    text-align:center;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:50px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ----------------------------
# Header
# ----------------------------

st.markdown('<p class="title">📧 AI Spam Email Classifier</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Machine Learning powered email & SMS spam detection system</p>',
    unsafe_allow_html=True
)

# ----------------------------
# Layout
# ----------------------------

left, right = st.columns([2,1])

with left:

    message = st.text_area(
        "✉️ Enter Email or SMS",
        height=220,
        placeholder="Example:\nCongratulations! You have won ₹50,000. Click here to claim your prize."
    )

    col1, col2 = st.columns(2)

    predict = col1.button("🔍 Predict", use_container_width=True)
    clear = col2.button("🗑 Clear", use_container_width=True)

if predict:

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        with st.spinner("Analyzing message..."):
            time.sleep(2)

            vector = vectorizer.transform([message])

            prediction = model.predict(vector)[0]

            probabilities = model.predict_proba(vector)[0]

        st.divider()

        if prediction == 1:

            st.error("🚨 This message is SPAM")

            st.progress(float(probabilities[1]))

            st.metric(
                label="Spam Confidence",
                value=f"{probabilities[1]*100:.2f}%"
            )

        else:

            st.success("✅ This message is NOT SPAM")

            st.progress(float(probabilities[0]))

            st.metric(
                label="Safe Confidence",
                value=f"{probabilities[0]*100:.2f}%"
            )

with right:

    st.subheader("📊 Project Information")

    st.info("""
Model Used:
- Multinomial Naive Bayes

Vectorizer:
- TF-IDF

Language:
- Python

Framework:
- Streamlit

Machine Learning:
- Scikit-learn
""")

    st.subheader("💡 Sample Messages")

    if st.button("Load Spam Example"):

        st.session_state["sample"] = "Congratulations! You have won ₹100000. Click the link to claim your prize now."

    if st.button("Load Safe Example"):

        st.session_state["sample"] = "Hi Sachin, let's meet tomorrow at 10 AM in the college library."

if "sample" in st.session_state:
    st.text_area(
        "Example Message",
        st.session_state["sample"],
        height=120
    )

st.markdown("---")

st.markdown(
"""
<div class="footer">
Made with ❤️ using Python, Scikit-Learn & Streamlit
</div>
""",
unsafe_allow_html=True
)