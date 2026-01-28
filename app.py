import streamlit as st
import pandas as pd
import base64
import io
import json
from openai import OpenAI

st.set_page_config(page_title="Business Card Extractor", layout="wide")
st.title("📇 Business Card Info Extractor")

# --- Sidebar ---
st.sidebar.header("🔑 OpenAI Settings")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

uploaded_files = st.sidebar.file_uploader(
    "Upload Business Card Images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- Extraction Function ---
def extract_business_card_info(image_bytes):
    image_base64 = base64.b64encode(image_bytes).decode()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Extract structured contact and social media information from business cards."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL business card information and return VALID JSON with these fields:\n\n"
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
                            "If a field is missing, return an empty string."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content

# --- Main ---
data_rows = []

if uploaded_files:
    st.subheader("📄 Extracted Data")

    for file in uploaded_files:
        with st.spinner(f"Processing {file.name}..."):
            image_bytes = file.read()
            result = extract_business_card_info(image_bytes)

            try:
                parsed = json.loads(result)
            except Exception as e:
                st.error(f"Failed to parse JSON from {file.name}")
                st.code(result)
                continue

            parsed["source_file"] = file.name
            data_rows.append(parsed)

            st.markdown(f"### {file.name}")
            st.json(parsed)

# --- CSV Export ---
if data_rows:
    df = pd.DataFrame(data_rows)

    st.subheader("⬇️ Download CSV")
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    st.download_button(
        label="Download Business Cards CSV",
        data=csv_buffer.getvalue(),
        file_name="business_cards_with_socials.csv",
        mime="text/csv"
    )
