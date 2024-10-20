import streamlit as st
import os
from PIL import Image

def get_image_files(image_dir):
    """Get all image files from the specified directory."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    return [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in image_extensions]

def display_gallery():
    st.title("Image Gallery")
    
    image_dir = "images"
    image_files = get_image_files(image_dir)
    
    if not image_files:
        st.write("No images found in the gallery.")
        return
    
    # Order images by most recent (assuming file names contain timestamps or are sorted by modification time)
    image_files = sorted(image_files, key=lambda x: os.path.getmtime(os.path.join(image_dir, x)), reverse=True)
    
    # Create a grid layout
    cols = st.columns(3)
    
    for idx, image_file in enumerate(image_files):
        with cols[idx % 3]:
            image_path = os.path.join(image_dir, image_file)
            img = Image.open(image_path)
            st.image(img, caption=image_file, use_column_width=True)
            
            # Add download button for each image
            with open(image_path, "rb") as file:
                btn = st.download_button(
                    label="Download",
                    data=file,
                    file_name=image_file,
                    mime="image/png"
                )

if __name__ == "__main__":
    st.set_page_config(page_title="MiniSwarm Gallery", page_icon="🖼️", layout="wide")
    display_gallery()