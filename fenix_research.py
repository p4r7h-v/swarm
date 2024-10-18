from swarm import Agent, Swarm
import os
import requests
import base64
from bs4 import BeautifulSoup
from openai import OpenAI
from PIL import Image
from clipboard import get_clipboard
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import play
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
from swarm.repl import run_demo_loop
from pandasql import sqldf
import datetime
import streamlit as st

# Initialize OpenAI client
openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

swarm_client = Swarm()

# Set up directories
data_dir = 'data/'
image_dir_name = "images"
image_dir = os.path.join(os.curdir, image_dir_name)

# Ensure image directory exists
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

# Check for existing index or create a new one
print("Checking if storage already exists...")
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    print("Loading documents and creating the index...")
    documents = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    print("Storing the index for later...")
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    print("Loading the existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Initialize chat engine
try:
    llm = LlamaOpenAI(model="gpt-4o")
    chat_engine = index.as_chat_engine(
        llm=llm,
        chat_mode="condense_plus_context",
        memory_buffer_size=100000,
        system_prompt=(
            "You are a helpful AI assistant. You are here to assist with any questions "
            "you may have about the documents. You end each response with a numbered "
            "list of follow-up actions, predicting what the user will ask next. You are "
            "not a search engine, and you do not have access to the internet."
        )
    )
    print("Chat engine initialized successfully.")
except Exception as e:
    print(f"Error initializing chat engine: {e}")

# Define tools (functions)
def ragbot(query: str) -> tuple:
    """Uses RAG to get possible results for a query based on the documents."""
    try:
        response = chat_engine.chat(query)
        # print("Response: ", str(response))
        # print("Sources: ", str(response.sources[0:30]))
        return "Response: " + str(response), "Sources: " + str(response.sources[0:30])
    except Exception as e:
        return f"Failed to answer query: {e}", None

def list_files(dir_path: str) -> str:
    """Lists the files in a directory."""
    try:
        return f'Files in {dir_path}:\n' + '\n'.join(os.listdir(dir_path))
    except Exception as e:
        return f"Failed to list files in {dir_path}: {e}"

def web_search(query: str) -> str:
    """Performs a web search using Bing and returns the top search results."""
    try:
        subscription_key = os.getenv("BING_API_KEY")
        assert subscription_key, "Bing Search key not found in environment variables"
        search_url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": subscription_key}
        params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
        return search_results
    except Exception as e:
        return f"Failed to perform web search: {e}"

def scrape_website(url: str, depth: int = 2) -> str:
    """Scrapes the content of a website up to a specified depth."""
    def scrape(url: str, current_depth: int) -> str:
        if current_depth > depth:
            return ""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'lxml')  # Use 'lxml' for HTML parsing
            text = soup.get_text()
            if current_depth < depth:
                links = [a['href'] for a in soup.find_all('a', href=True)]
                for link in links:
                    if link.startswith('http'):
                        text += scrape(link, current_depth + 1)
            return text
        except Exception as e:
            return f"Failed to scrape website '{url}': {e}"
    return scrape(url, 1)

def create_file(file_path: str, content: str) -> str:
    """Creates a file with the given content."""
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return f"File {file_path} created successfully."
    except Exception as e:
        return f"Failed to create file {file_path}: {e}"

def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """Reads the content of a file."""
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            return file.read()
    except Exception as e:
        return f"Failed to read file {file_path}: {e}"

def delete_file(file_path: str, require_permission: bool = True) -> str:
    """Deletes a file, requiring permission if specified."""
    try:
        if require_permission:
            # Ask for user confirmation
            confirmation = input(f"Are you sure you want to delete {file_path}? (yes/no): ").strip().lower()
            if confirmation != 'yes':
                return f"Deletion of {file_path} cancelled by user."

        os.remove(file_path)
        return f"File {file_path} deleted successfully."
    except Exception as e:
        return f"Failed to delete file {file_path}: {e}"

def rename_file(file_path: str, new_file_path: str) -> str:
    """Renames a file."""
    try:
        os.rename(file_path, new_file_path)
        return f"File {file_path} renamed to {new_file_path} successfully."
    except Exception as e:
        return f"Failed to rename file {file_path} to {new_file_path}: {e}"

def copy_file(file_path: str, new_file_path: str) -> str:
    """Copies a file."""
    try:
        with open(file_path, 'rb') as file:
            content = file.read()
        with open(new_file_path, 'wb') as new_file:
            new_file.write(content)
        return f"File {file_path} copied to {new_file_path} successfully."
    except Exception as e:
        return f"Failed to copy file {file_path} to {new_file_path}: {e}"

def move_file(file_path: str, new_file_path: str) -> str:
    """Moves a file to a new location."""
    try:
        os.replace(file_path, new_file_path)
        return f"File {file_path} moved to {new_file_path} successfully."
    except Exception as e:
        return f"Failed to move file {file_path} to {new_file_path}: {e}"

def append_to_file(file_path: str, content: str) -> str:
    """Appends content to an existing file."""
    try:
        with open(file_path, 'a') as file:
            file.write(content)
        return f"Content appended to {file_path} successfully."
    except Exception as e:
        return f"Failed to append content to {file_path}: {e}"

def rewrite_section(file_path: str, section_title: str, new_content: str) -> str:
    """Rewrites a specific section in the file identified by the section title."""
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        lines = content.split('\n')
        start_idx = -1
        end_idx = -1

        # Find the start of the section
        for i, line in enumerate(lines):
            if section_title in line:
                start_idx = i
                break

        if start_idx == -1:
            return f"Section titled '{section_title}' not found in {file_path}."

        # Find the end of the section
        for i in range(start_idx + 1, len(lines)):
            # Check for the next section or end of file
            if lines[i].strip() and (lines[i].strip().startswith('##') or lines[i].strip().startswith('#')):
                end_idx = i
                break

        if end_idx == -1:
            end_idx = len(lines)

        # Replace the section content
        modified_lines = lines[:start_idx + 1] + new_content.split('\n') + lines[end_idx:]
        modified_content = '\n'.join(modified_lines)

        with open(file_path, 'w') as file:
            file.write(modified_content)

        return f"Section '{section_title}' rewritten successfully in {file_path}."
    except Exception as e:
        return f"Failed to rewrite section in {file_path}: {e}"

def get_folder_structure(root_dir, indent='  '):
    """
    Generates a token-efficient representation of a folder's file names.   

    Parameters:
    - root_dir (str): The root directory to scan.
    - indent (str): The string used for indenting nested files and directories.

    Returns:
    - str: A string representing the folder structure.
    """
    folder_structure = []

    for root, dirs, files in os.walk(root_dir):
        # Calculate the current level of indentation
        level = root.replace(root_dir, '').count(os.sep)
        current_indent = indent * level

        # Add the directory name
        folder_name = os.path.basename(root)
        folder_structure.append(f"{current_indent}{folder_name}/")

        # Add the files in this directory
        for file in files:
            folder_structure.append(f"{current_indent}{indent}{file}")     

    return '\n'.join(folder_structure)

def check_file_existence(file_path: str) -> bool:
    """Checks if a file exists at the given path."""
    try:
        return os.path.exists(file_path)
    except Exception as e:
        return f"Failed to check existence of file {file_path}: {e}"

def get_file_metadata(file_path: str) -> dict:
    """Retrieves metadata such as size, creation date, and modification date of a file."""
    try:
        metadata = os.stat(file_path)
        return {
            "size": metadata.st_size,
            "creation_date": metadata.st_ctime,
            "modification_date": metadata.st_mtime
        }
    except Exception as e:
        return {"error": str(e)}

def generate_image(prompt: str, image_filepath: str, model: str = "dall-e-3", n: int = 1, size: str = "1024x1024", response_format: str = "url", quality: str = "standard") -> str:
    """Generates an image based on a text prompt using DALL-E. Size must be one of the following: 1024x1024, 1792x1024, or 1024x1792. Quality must be one of the following: standard, hd."""
    try:
        generation_response = openai.images.generate(
            model=model,
            prompt=prompt,
            n=n,
            quality=quality,
            size=size,
            response_format=response_format,
        )
        generated_image_url = generation_response.data[0].url
        generated_image = requests.get(generated_image_url).content
        with open("images/" + image_filepath, "wb") as image_file:
            image_file.write(generated_image)
        return "images/" + image_filepath
    except Exception as e:
        return f"Failed to generate image: {e}"

def display_image(image_path: str):
    """Displays an image from the given file path."""
    try:
        image = Image.open(image_path)
        image.show()
        return f"Image displayed."
    except Exception as e:
        return f"Failed to display image: {e}"

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
            "max_tokens": max_tokens
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()
    except Exception as e:
        return f"Failed to analyze image: {e}"

def create_folder(folder_path: str) -> str:
    """Creates a folder at the specified path."""
    try:
        os.makedirs(folder_path)
        return f"Folder {folder_path} created successfully."
    except Exception as e:
        return f"Failed to create folder {folder_path}: {e}"
    
import pygame

def synthesize_speech(text: str, voice: str = "alloy", output_file: str = "output.mp3") -> str:
    """Synthesizes speech from text using OpenAI's TTS model and plays the audio."""
    try:
        # Initialize the OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Create the speech using the OpenAI Audio API
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        
        # Define the path to save the speech file
        speech_file_path = Path(output_file)

        # Save the speech file
        with open(speech_file_path, "wb") as file:
            file.write(response.content)
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Play the audio file
        pygame.mixer.music.load(speech_file_path)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        return f"Speech synthesized and saved as {speech_file_path}."
    except Exception as e:
        return f"Failed to synthesize speech: {e}"
    
def generate_streaming_voice(text: str) -> str:
    """Generates and streams a voice using ElevenLabs."""
    try:
        client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        
        audio_stream = client.generate(
            text=text,
            stream=True
        )
        
        stream(audio_stream)
        
        return "Voice generated and streamed successfully."
    except Exception as e:
        return f"Failed to generate and stream voice: {e}"

# Define functions for Data Analysis Agent
def clean_data(data: str) -> dict:
    """Cleans the provided data by removing duplicates."""
    data_io = StringIO(data)
    df = pd.read_csv(data_io, sep=",")
    df_deduplicated = df.drop_duplicates()
    return {"cleaned_data": df_deduplicated.to_dict()}

def stat_analysis(data: str) -> dict:
    """Performs statistical analysis on the given dataset."""
    data_io = StringIO(data)
    df = pd.read_csv(data_io, sep=",")
    return {"stats": df.describe().to_dict()}

def plot_line_chart(data: str) -> None:
    """Creates a line chart from the provided data."""
    data_io = StringIO(data)
    df = pd.read_csv(data_io, sep=",")
    
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    y_fit = polynomial(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label='Data Points')
    plt.plot(x, y_fit, '-', label='Best Fit Line')
    plt.title('Line Chart with Best Fit Line')
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_bar_chart(data: str) -> None:
    """Creates a bar chart from the provided data."""
    data_io = StringIO(data)
    df = pd.read_csv(data_io, sep=",")
    
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    
    plt.figure(figsize=(10, 6)) 
    plt.bar(x, y)
    plt.title('Bar Chart')
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.grid(True)
    plt.show()

def plot_histogram(data: str) -> None:  
    """Creates a histogram from the provided data. Pass in a string of comma separated values for x."""
    data_io = StringIO(data)
    df = pd.read_csv(data_io, sep=",")
    
    plt.figure(figsize=(10, 6))
    plt.hist(df.iloc[:, 0], bins=10)
    plt.title('Histogram')
    plt.xlabel(df.columns[0])   
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_scatter_plot(data: str) -> None:
    """Creates a scatter plot from the provided data."""
    data_io = StringIO(data)
    df = pd.read_csv(data_io, sep=",")  
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
    plt.title('Scatter Plot')
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.grid(True)
    plt.show()

def plot_pie_chart(data: str) -> None:
    """Creates a pie chart from the provided data."""
    data_io = StringIO(data)
    df = pd.read_csv(data_io, sep=",")
    
    plt.figure(figsize=(10, 6))
    plt.pie(df.iloc[:, 1], labels=df.iloc[:, 0], autopct='%1.1f%%')
    plt.title('Pie Chart')
    plt.show()      


def add_numbers(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b

def subtract_numbers(a: float, b: float) -> float:
    """Subtracts two numbers."""
    return a - b

def multiply_numbers(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b

def divide_numbers(a: float, b: float):
    """Divides two numbers."""
    if b == 0:
        return "Error: Division by zero is not allowed."
    return a / b

def linkedin_search(prompt: str):
    """Execute a SQL query on LinkedIn data stored in a CSV file.
    
    Column names: First Name, Last Name, URL, Email Address, Company, Position, Connected On
    
    Examples of SQL queries you can use:
    - `SELECT * FROM df WHERE Company = 'OpenAI';`
    - `SELECT `First Name`, `Last Name` FROM df WHERE `Connected On` > '2023-01-01';`
    - `SELECT Company, COUNT(*) AS ContactCount FROM df GROUP BY Company ORDER BY ContactCount DESC;`
    
    Use 'df' as the table name and enclose column names with spaces in backticks (`).
    """
    file_path = 'LinkedIn.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found. You can download your Connections data from your LinkedIn profile by going to 'Settings' > 'Data and Privacy' > 'Get a copy of your data' > 'Connections' > 'Download Archive'."
    except Exception as e:
        return f"An error occurred: {e}"
    
    try:
        print("Prompt:", prompt)
        result = sqldf(prompt, {'df': df})
        if result.empty:
            return "No results found."
        return result.to_string(index=False)
    except Exception as e:
        return f"Error executing query: {e}"
    
# Define transfer functions
def transfer_back_to_triage():
    """Transfers back to the Triage Agent."""
    return triage_agent

def transfer_to_research():
    """Transfers to the Research Agent."""
    return research_agent

def transfer_to_file_management():
    """Transfers to the File Management Agent."""
    return file_management_agent

def transfer_to_image_processing():
    """Transfers to the Image Processing Agent."""
    return image_processing_agent

def transfer_to_codebase():
    """Transfers to the Codebase Agent."""
    return codebase_agent

def transfer_to_voice():
    """Transfers to the Voice Agent."""
    return voice_agent

def transfer_to_data_analysis():
    """Transfers to the Data Analysis Agent."""
    return data_analysis_agent

def transfer_to_math():
    """Transfers to the Math Agent."""
    return math_agent

def transfer_to_linkedin():
    """Transfers to the LinkedIn Agent."""
    return linkedin_agent

# Define agents for Fenix Research
triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        f"You are the Triage Agent for Fenix Research. You are CLI based, so don't use markdown for chat. The Fenix Research program is capable of handling various tasks through specialized agents. Determine which agent is best suited to handle the user's request, and transfer the conversation to that agent. The agents available are: Research Agent for web searches and document retrieval, File Management Agent for managing files and directories, Image Processing Agent for image generation and processing, Codebase Agent for managing and analyzing the codebase, Voice Agent for text-to-speech synthesis, Data Analysis Agent for data cleaning, statistical analysis, and line chart generation, Math Agent for basic arithmetic operations, LinkedIn Agent for executing SQL queries on LinkedIn data stored in a CSV file. Today's date is {datetime.date.today()}."
    ),
    functions=[
        transfer_to_research,
        transfer_to_file_management,
        transfer_to_image_processing,
        transfer_to_codebase,
        transfer_to_voice,
        transfer_to_data_analysis,
        transfer_to_math,
        transfer_to_linkedin
    ]
)

research_agent = Agent(
    name="Research Agent",
    instructions=(
        f"You are the Research Agent. Handle research queries by performing web searches, scraping websites, and retrieving answers from documents. Today's date is {datetime.date.today()}."
    ),
    functions=[web_search, scrape_website, ragbot, transfer_back_to_triage]
)

file_management_agent = Agent(
    name="File Management Agent",
    instructions=(
        "You are the File Management Agent. Manage files and directories on the system."
    ),
    functions=[
        create_file, read_file, delete_file, rename_file, copy_file, move_file,
        append_to_file, check_file_existence, get_file_metadata, transfer_back_to_triage
    ]
)

image_processing_agent = Agent(
    name="Image Processing Agent",
    instructions=(
        "You are the Image Processing Agent. Handle image generation and processing requests."
    ),
    functions=[generate_image, display_image, analyze_image, transfer_back_to_triage]
)

codebase_agent = Agent(
    name="Codebase Agent",
    instructions=(
        "You are the Codebase Agent. Manage and analyze the codebase."
    ),
    functions=[read_file, get_file_metadata, transfer_back_to_triage]
)

voice_agent = Agent(
    name="Voice Agent",
    instructions=(
        "You are the Voice Agent. Use synthesis to convert text to speech."
    ),
    functions=[synthesize_speech, generate_streaming_voice, transfer_back_to_triage]
)

data_analysis_agent = Agent(
    name="Data Analysis Agent",
    instructions=(
        "You are the Data Analysis Agent. Perform data analysis tasks including cleaning data, conducting statistical analysis, and generating line charts."
    ),
    functions=[clean_data, stat_analysis, plot_line_chart, plot_bar_chart, plot_histogram, plot_scatter_plot, plot_pie_chart, transfer_back_to_triage]
)

# Define Math Agent
math_agent = Agent(
    name="Math Agent",
    instructions=(
        "You are the Math Agent. Perform basic arithmetic operations including addition, subtraction, multiplication, and division."
    ),
    functions=[
        add_numbers,
        subtract_numbers,
        multiply_numbers,
        divide_numbers,
        transfer_back_to_triage
    ]
)

linkedin_agent = Agent(
    name="LinkedIn Agent",
    instructions=(
        "You are the LinkedIn Agent. Use the 'linkedin_search' function to execute SQL queries on LinkedIn data stored in a CSV file."
    ),
    functions=[linkedin_search, transfer_back_to_triage]
)

# Define avatar mapping for agents
agent_avatars = {
    "user": "user.png",
    "assistant": "neon_phoenix_icon_variant3.png"
}

def preprocess(chunk):
    content = ""
    sender = None

    if "sender" in chunk and chunk["sender"]:
        sender = chunk["sender"]
        st.session_state["agent"] = agent_mapping[sender]
    if "content" in chunk and chunk["content"]:
        content_chunk = chunk["content"]
        content += content_chunk

    if "tool_calls" in chunk and chunk["tool_calls"]:
        for tool_call in chunk["tool_calls"]:
            f = tool_call["function"]
            name = f.get("name")
            if name and name != "":
                content += f"`{name}()`\n\n"  # Use markdown for code formatting


    if "delim" in chunk and chunk["delim"] == "end" and content:
        content += "\n"  # End of response message

    return content, sender

# Modify the process_stream function
def process_stream(stream):
    assistant_content = ""
    image_path = None
    for chunk in stream:
        content, sender = preprocess(chunk)
        if sender and content:
            assistant_content += f"**{sender}:** {content}\n"
        elif content:
            assistant_content += content
        
        # Check if the content is an image path
        if "Image generated and saved to " in content:
            image_path = content.split("Image generated and saved to ")[-1].strip(".")
        
        yield assistant_content, image_path

    # After processing all chunks, append the assistant's message to session state
    if assistant_content:
        message = {"role": "assistant", "content": assistant_content.strip()}
        if image_path:
            message["image"] = image_path
        st.session_state["messages"].append(message)

#  agent mapping
agent_mapping = {
    "Triage Agent": triage_agent,
    "Research Agent": research_agent,
    "File Management Agent": file_management_agent,
    "Image Processing Agent": image_processing_agent,
    "Codebase Agent": codebase_agent,
    "Voice Agent": voice_agent,
    "Data Analysis Agent": data_analysis_agent,
    "Math Agent": math_agent,
    "LinkedIn Agent": linkedin_agent
}


# Initialize session state to store messages and the agent
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "agent" not in st.session_state:
    st.session_state["agent"] = agent_mapping["Triage Agent"]


# Title of the chat app
st.title("MiniSwarm.AI")

# Display previous chat messages
for message in st.session_state["messages"]:
    role = message.get("role", "assistant")
    avatar = agent_avatars.get(role, "🤖")  # Default to 🤖 if role not found
    # Use st.chat_message without avatar if it's an emoji
    if len(avatar) == 1 or avatar.startswith('🧑‍'):  # Check if it's an emoji
        with st.chat_message(role):
            st.markdown(message["content"])
            if "image" in message:
                st.image(message["image"])
    else:
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])
            if "image" in message:
                st.image(message["image"])

# User input through Streamlit chat input
user_input = st.chat_input("Type your message")

if user_input:
    # Append user's message to session state and display it
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # Display user message in chat
    with st.chat_message("user", avatar=agent_avatars["user"]):
        st.markdown(user_input)

    # Send user input to Swarm agent for processing
    response = swarm_client.run(
        agent=st.session_state["agent"],
        messages=st.session_state["messages"],
        context_variables={},  # Add any necessary context here
        stream=True,  # Optional: you can enable streaming responses
        debug=False
    )

    # Display Swarm's response in chat
    with st.chat_message("assistant", avatar=agent_avatars["assistant"]):
        message_placeholder = st.empty()
        for content, _ in process_stream(response):
            message_placeholder.markdown(content)