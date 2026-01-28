import streamlit as st
import pandas as pd
import base64
import io
import json
from PIL import Image
import pillow_heif
from openai import OpenAI

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Business Card Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📇 Business Card Info Extractor")
st.caption("Upload business cards (PNG, JPG, HEIC). Extract contact + social info. Export CSV.")

# ------------------ SIDEBAR ------------------
st.sidebar.header("🔑 OpenAI Settings")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Required for vision-based extraction"
)

uploaded_files = st.sidebar.file_uploader(
    "Upload Business Card Images",
    type=["png", "jpg", "jpeg", "heic", "heif"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.caption("Built for Streamlit Community Cloud")

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

client = OpenAI(api_key=api_key)

# ------------------ IMAGE HANDLING ------------------
def load_and_convert_image(uploaded_file):
    suffix = uploaded_file.name.split(".")[-1].lower()

    if suffix in ["heic", "heif"]:
        heif_file = pillow_heif.read_heif(uploaded_file)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw"
        )
    else:
        image = Image.open(uploaded_file)

    image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)

    return buffer.read()

# ------------------ OPENAI EXTRACTION ------------------
def extract_business_card_info(image_bytes):
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You extract structured contact and social media information from business cards."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL information from this business card and return VALID JSON ONLY.\n\n"
                            "Fields:\n"
                            "name\n"
                            "title\n"
                            "company\n"
                            "email\n"
                            "phone\n"
                            "website\n"
                            "address\n"
                            "linkedin\n"
                            "twitter\n"
                            "instagram\n"
                            "facebook\n"
                            "youtube\n"
                            "tiktok\n"
                            "other_socials\n"
                            "notes\n\n"
                            "If missing, return empty string. No explanations."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content

# ------------------ MAIN LOGIC ------------------
rows = []

if uploaded_files:
    st.subheader("📄 Extracted Results")

    for file in uploaded_files:
        with st.spinner(f"Processing {file.name}..."):
            try:
                image_bytes = load_and_convert_image(file)
                result = extract_business_card_info(image_bytes)
                parsed = json.loads(result)

                parsed["source_file"] = file.name
                rows.append(parsed)

                st.markdown(f"### {file.name}")
                st.json(parsed)

            except Exception as e:
                st.error(f"Failed to process {file.name}")
                st.code(str(e))

# ------------------ CSV EXPORT ------------------
if rows:
    df = pd.DataFrame(rows)

    st.subheader("⬇️ Download CSV")
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="Download Business Cards CSV",
        data=csv_buffer.getvalue(),
        file_name="business_cards_extracted.csv",
        mime="text/csv"
    )
