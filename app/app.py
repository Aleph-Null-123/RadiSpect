import pandas as pd
import streamlit as st
import matplotlib.image as mpimg
from pathlib import Path

st.set_page_config(page_title="RadiSpect — Viewer", layout="wide")

@st.cache_data(show_spinner=False)
def load_pairs(csv_path: str):
    df = pd.read_csv(csv_path)
    # be strict: only image + report columns
    df = df[["image", "report"]].copy()
    return df

pairs_path = "data/pairs.csv"
if not Path(pairs_path).exists():
    st.error("data/pairs.csv not found. Run Step 1 to generate it.")
    st.stop()

df = load_pairs(pairs_path)
st.sidebar.title("RadiSpect")
st.sidebar.write(f"Loaded **{len(df)}** pairs")

# Simple filter box for quick narrowing
q = st.sidebar.text_input("Filter filenames (contains):", "")
subset = df[df["image"].str.contains(q, case=False, na=False)] if q else df

if subset.empty:
    st.warning("No matches. Clear the filter to see all images.")
    st.stop()

# Selection UI
default_index = 0
img_path = st.sidebar.selectbox(
    "Choose image",
    subset["image"].tolist(),
    index=default_index,
)

# Main panel: render image (grayscale pngs load fine via mpimg)
try:
    arr = mpimg.imread(img_path)
except Exception as e:
    st.error(f"Failed to load image: {img_path}\n{e}")
    st.stop()

st.subheader(Path(img_path).name)
st.image(arr, use_container_width=True, clamp=True)

with st.expander("Report (for reference)"):
    st.write(subset.loc[subset["image"] == img_path, "report"].iloc[0])
