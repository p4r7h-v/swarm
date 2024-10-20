import time
from runwayml import RunwayML
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions, ContentSettings
import os
import requests
import base64
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import sys
import re

load_dotenv()
# The env var RUNWAYML_API_SECRET is expected to contain your API key.
client = RunwayML(
    # default is 2
    max_retries=0,
)

# Load your Azure Storage connection string from an environment variable or provide it directly
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING', '<your_connection_string>')

def analyze_image(image_path: str, prompt: str, max_tokens: int = 300) -> str:
    """Analyzes a local image using GPT-4 with vision capabilities."""
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '<your OpenAI API key if not set as env var>')}"
        }
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 250
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()
    except Exception as e:
        return f"Failed to analyze image: {e}"

def generate_video(image_path: str, prompt: str) -> str:
    """
    Generates a video based on the provided image and prompt.

    Parameters:
    - image_path (str): The file path to the input image.
    - prompt (str): The prompt to guide video generation.

    Returns:
    - str: SAS URL of the generated video or an error message.
    """
    try:
        # Initialize Azure Blob Storage client
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        container_name = 'images'
        
        # Upload image to Azure Blob Storage
        blob_name = os.path.basename(image_path)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        # Determine Content-Type based on file extension
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension == '.png':
            content_type = 'image/png'
        elif file_extension in ['.jpg', '.jpeg']:
            content_type = 'image/jpeg'
        else:
            content_type = 'application/octet-stream'  # Default fallback
        
        with open(image_path, 'rb') as data:
            blob_client.upload_blob(
                data,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type)
            )
        
        # Generate SAS token for the uploaded image
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=1)  # Token valid for 1 hour
        )
        
        # Construct the SAS URL
        uploaded_image_url = f"{blob_client.url}?{sas_token}"
        print(f"SAS URL: {uploaded_image_url}")
        
        # Analyze the image to create an interesting prompt for the animation
        prompt_text = 'Write a prompt for a 10 second video beginning with this image. The prompt will be used to generate the video using a video generation model.'
        analysis_result = analyze_image(image_path, prompt_text)
        
        while isinstance(analysis_result, dict) and len(analysis_result.get('choices', [{}])[0].get('message', {}).get('content', '')) >= 512:
            prompt_text = analysis_result['choices'][0]['message']['content']
            analysis_result = analyze_image(image_path, f'Shorten this prompt: {prompt_text}')
        
        if isinstance(analysis_result, dict):
            final_prompt = analysis_result['choices'][0]['message']['content']
            print(final_prompt)
        else:
            final_prompt = analysis_result
            print(final_prompt)
        
        # Generate a timestamp-based file name for the animation
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        final_file_name = f"{timestamp}.mp4"
        print(f"Generated Filename: {final_file_name}")
        
        # Create a new image-to-video task using the "gen3a_turbo" model
        task = client.image_to_video.create(
          model='gen3a_turbo',
          # Point this at your own image file
          prompt_image=uploaded_image_url,
          prompt_text=final_prompt,
        )
        task_id = task.id
        
        # Poll the task until it's complete
        time.sleep(10)  # Wait for ten seconds before polling
        task = client.tasks.retrieve(task_id)
        while task.status not in ['SUCCEEDED', 'FAILED']:
            time.sleep(10)  # Wait for ten seconds before polling
            task = client.tasks.retrieve(task_id)
        
        if task.status != 'SUCCEEDED':
            return f"Video generation failed with status: {task.status}"
        
        print('Task complete:', task)
        
        # Download the generated video
        video_url = task.output[0]
        video_response = requests.get(video_url)
        
        # Save the video locally
        local_video_path = f"videos/{final_file_name}"
        os.makedirs(os.path.dirname(local_video_path), exist_ok=True)
        with open(local_video_path, 'wb') as video_file:
            video_file.write(video_response.content)
        
        # Upload the video to Azure Blob Storage
        video_blob_name = os.path.basename(local_video_path)
        video_blob_client = blob_service_client.get_blob_client(container=container_name, blob=video_blob_name)
        
        with open(local_video_path, 'rb') as video_file:
            video_blob_client.upload_blob(video_file, overwrite=True, content_settings=ContentSettings(content_type='video/mp4'))
        
        # Generate SAS token for the uploaded video
        video_sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=container_name,
            blob_name=video_blob_name,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=1)  # Token valid for 1 hour
        )
        
        # Construct the SAS URL for the video
        uploaded_video_url = f"{video_blob_client.url}?{video_sas_token}"
        print(f"Video SAS URL: {uploaded_video_url}")
        
        return uploaded_video_url
    
    except Exception as e:
        return f"Failed to generate video: {e}"

# Example usage:
if __name__ == "__main__":
    video_url = generate_video('firefly.png', 'slow motion')
    print(f"Generated Video URL: {video_url}")
