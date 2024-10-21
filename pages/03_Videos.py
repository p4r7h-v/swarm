import streamlit as st
import os

def get_video_files(video_dir):
    """Get all video files from the specified directory."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    return [f for f in os.listdir(video_dir) if os.path.splitext(f)[1].lower() in video_extensions]

def display_video_gallery():
    st.title("Video Gallery")
    
    video_dir = "videos/"
    video_files = get_video_files(video_dir)
    
    if not video_files:
        st.write("No videos found in the gallery.")
        return
    
    # Order videos by most recent
    video_files = sorted(video_files, key=lambda x: os.path.getmtime(os.path.join(video_dir, x)), reverse=True)
    
    # Create a grid layout
    cols = st.columns(2)
    
    for idx, video_file in enumerate(video_files):
        with cols[idx % 2]:
            video_path = os.path.join(video_dir, video_file)
            st.video(video_path, format="video/mp4")
            st.caption(video_file)
            
            # Add download button for each video
            with open(video_path, "rb") as file:
                btn = st.download_button(
                    label="Download",
                    data=file,
                    file_name=video_file,
                    mime="video/mp4"
                )

if __name__ == "__main__":
    st.set_page_config(page_title="MiniSwarm Video Gallery", page_icon="🎬", layout="wide")
    display_video_gallery()