import streamlit as st
import pandas as pd
import base64
import io
import json
from PIL import Image
import pillow_heif
from openai import OpenAI

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Business Card Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📇 Business Card Extractor")
st.caption("Upload business cards → extract contact + social info → download CSV")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("🔑 OpenAI Configuration")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Required for image-based extraction"
)

uploaded_files = st.sidebar.file_uploader(
    "Upload Business Cards",
    type=["png", "jpg", "jpeg", "heic", "heif"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.caption("Optimized for Streamlit Community Cloud")

if not api_key:
    st.warning("Enter your OpenAI API key to continue.")
    st.stop()

client = OpenAI(api_key=api_key)

# --------------------------------------------------
# IMAGE LOADING & CONVERSION
# --------------------------------------------------
def load_and_convert_image(file):
    ext = file.name.split(".")[-1].lower()

    try:
        if ext in ["heic", "heif"]:
            heif = pillow_heif.read_heif(file)
            image = Image.frombytes(
                heif.mode,
                heif.size,
                heif.data,
                "raw"
            )
        else:
            image = Image.open(file)

        image = image.convert("RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)

        return buffer.read()

    except Exception as e:
        raise RuntimeError(f"Image conversion failed: {e}")

# --------------------------------------------------
# OPENAI EXTRACTION (FAIL-SAFE)
# --------------------------------------------------
def extract_business_card_info(image_bytes, retries=1):
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract business card data. "
                            "Return ONLY valid JSON. "
                            "No markdown. No commentary."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Extract ALL information and return JSON ONLY.\n\n"
                                    "Fields:\n"
                                    "name, title, company, email, phone, website, address,\n"
                                    "linkedin, twitter, instagram, facebook, youtube, tiktok,\n"
                                    "other_socials, notes\n\n"
                                    "If missing, return empty string."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ]
            )

            raw = response.choices[0].message.content.strip()

            # Remove markdown if model disobeys
            if raw.startswith("```"):
                raw = raw.replace("```json", "").replace("```", "").strip()

            return json.loads(raw)

        except Exception:
            if attempt < retries:
                continue

            # Hard fallback — never crash
            return {
                "name": "",
                "title": "",
                "company": "",
                "email": "",
                "phone": "",
                "website": "",
                "address": "",
                "linkedin": "",
                "twitter": "",
                "instagram": "",
                "facebook": "",
                "youtube": "",
                "tiktok": "",
                "other_socials": "",
                "notes": "Extraction failed or unreadable card"
            }

# --------------------------------------------------
# MAIN APP LOGIC
# --------------------------------------------------
rows = []

if uploaded_files:
    st.subheader("📄 Extraction Results")

    for file in uploaded_files:
        with st.spinner(f"Processing {file.name}..."):
            try:
                image_bytes = load_and_convert_image(file)
                data = extract_business_card_info(image_bytes, retries=1)

                data["source_file"] = file.name
                rows.append(data)

                st.markdown(f"### {file.name}")
                st.json(data)

            except Exception as e:
                st.error(f"Failed to process {file.name}")
                st.code(str(e))

# --------------------------------------------------
# CSV EXPORT
# --------------------------------------------------
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
