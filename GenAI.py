import streamlit as st
import google.generativeai as genai
import os
import PIL.Image
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Load Gemini model
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# ✅ Function to analyze human attributes
def analyze_human_attributes(uploaded_file):
    prompt = """
    You are an AI trained to analyze human attributes from images with high accuracy. 
    Carefully analyze the given image and return the following structured details:

    You have to return all results as you have the image, don't want any apologize or empty results.

    - **Gender** (Male/Female/Non-binary)
    - **Age Estimate**
    - **Ethnicity**
    - **Mood**
    - **Facial Expression**
    - **Glasses**
    - **Beard**
    - **Hair Color**
    - **Eye Color**
    - **Headwear**
    - **Emotions Detected**
    - **Confidence Level**
    """

    # ✅ Read image data as bytes
    image_bytes = uploaded_file.read()

    # ✅ Prepare image for Gemini
    image_data = {
        "inline_data": {
            "mime_type": uploaded_file.type,  # e.g., image/jpeg
            "data": image_bytes
        }
    }

    # ✅ Send prompt and image to Gemini
    response = model.generate_content([prompt, image_data])
    return response.text.strip()


# ✅ Streamlit App UI
st.set_page_config(page_title="Human Attribute Detector", layout="wide")
st.title("🧠 Human Attribute Detection")
st.write("Upload an image to detect human attributes using Google's Gemini AI.")

uploaded_image = st.file_uploader("📷 Upload an Image", type=['png', 'jpg', 'jpeg'])

if uploaded_image:
    # ✅ Display image
    img = PIL.Image.open(uploaded_image)

    # ✅ Reset stream to allow re-read
    uploaded_image.seek(0)

    # ✅ Analyze
    person_info = analyze_human_attributes(uploaded_image)

    # ✅ Show results side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.markdown("### 🧾 Detected Attributes")
        st.write(person_info)
