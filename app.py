import os
import streamlit as st
from PIL import Image
import openai
import base64
import torch
import io
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain.callbacks import FileCallbackHandler
from loguru import logger
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
import google.generativeai as genai
from torchvision import transforms
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from symptom_analysis import analyze_symptom
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from streamlit_chat import message
from langchain.callbacks import get_openai_callback, StreamingStdOutCallbackHandler, FileCallbackHandler
from langchain.schema import AIMessage, HumanMessage

open_api_key = "your api key"
# Function for chatbot responses
def generate_chatbot_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

# Custom CSS for removing top space, styling the page with Eagle Lake font, input boxes & buttons
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Eagle+Lake&display=swap');
    
    body {
        background-color: #f0f0f5;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Eagle Lake', cursive;
        color: #1a73e8;
        font-weight: 600;
        text-align: center;
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    img {
        width: 100%;
        max-height: 350px;
        object-fit: cover;
        border-radius: 8px;
    }

    .custom-input {
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 12px;
        width: 100%;
        margin-bottom: 20px;
        font-size: 16px;
        font-family: 'Arial', sans-serif;
    }
    
    .custom-button {
        background-color: #1a73e8;
        color: white;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 10px 2px;
        cursor: pointer;
        border-radius: 12px;
        border: none;
    }
    
    .custom-button:hover {
        background-color: #0066cc;
    }

    .no-top-margin {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    </style>
    """, unsafe_allow_html=True)

# Load section banner images
def load_image_as_base64(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def display_image_and_content(image_path, content):
    """Display image and content after the header."""
    st.markdown(f"""
        <img src="data:image/jpeg;base64,{load_image_as_base64(image_path)}" alt="Image">
    """, unsafe_allow_html=True)
    st.write(content)

# Paths for small images to be placed beside the headings
image_analysis_small = r"C:\Users\paill\Downloads\Image.jpg"
symptom_analysis_small = r"C:\Users\paill\Downloads\symptom.jpg"
chatbot_small = r"C:\Users\paill\Downloads\chatbot.jpg"
home_image = r"C:\Users\paill\Downloads\AI powered heathcare.jpg"

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a section", ["Home", "Symptom Analysis","Image Analysis", "Medical Chatbot"])

# Home Page
if app_mode == "Home":
    st.title("Welcome to AI Healthcare App")
    display_image_and_content(home_image, """
        This app provides several features like:
        - Image Analysis for medical images
        - Symptom Analysis based on input
        - Medical Chatbot to answer your health-related queries

        Use the sidebar to navigate through the features.
    """)

# API key configuration
API_KEY = "your key"
genai.configure(api_key=API_KEY)

# Define the system prompt
system_prompt = {
    "text": (
        "You are a highly skilled AI trained in advanced medical image analysis, specialized in reading and interpreting "
        "complex medical images such as MRIs, CT scans,X-rays and skin images. Your task is to perform the following steps:\n\n"
        
        "1. **Comprehensive Image Analysis:** Thoroughly examine the provided medical image. Focus on detecting any abnormalities, "
        "anomalies, signs of disease, or irregularities related to human health.\n"
        
        "2. **Detailed Findings Report:** Clearly document your observations in a structured format. Provide specific details "
        "about what was detected, including areas of concern, abnormalities, or patterns that may suggest a medical condition.\n"
        
        "3. **Anomalies and Diagnosis:** Highlight any critical anomalies or irregularities found in the image. Suggest potential "
        "diagnoses or conditions related to the findings, and mention any differential diagnoses that may need to be considered.\n"
        
        "4. **Recommendations and Next Steps:** Based on the image analysis, recommend potential next steps such as further diagnostic "
        "testing (e.g., biopsy, PET scan), specialist consultations, or other relevant medical procedures.\n"
        
        "5. **Treatment Suggestions:** If appropriate, propose treatment options or interventions based on the observed findings "
        "and possible diagnoses.\n\n"
        
        "Important Guidelines:\n"
        "- Only analyze the image if it pertains to human medical imaging, such as MRIs, CT scans, X-rays, or skin images.\n"
        "- If the image quality is insufficient for a proper analysis, notify the user that the findings may be inconclusive due to poor image quality.\n"
        "- Include this disclaimer with every analysis: 'Consult with a licensed medical professional before making any health-related decisions based on this analysis.'"
        "- provide the image type and body region of medical image"
    )
}

# Create the model with safety settings
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the GenerativeModel
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# **Symptom Analysis Page**
if app_mode == "Symptom Analysis":
    st.title("Symptom Analysis")
    display_image_and_content(symptom_analysis_small, " ")
    st.write("Enter your symptoms below, and the AI will analyze them.")

    # User input for symptoms
    symptoms = st.text_area("Enter symptoms (comma-separated):", placeholder="e.g., fever, headache, cough")

    if st.button("Analyze Symptoms"):
        if symptoms:
            with st.spinner("Analyzing symptoms..."):
                diagnosis = analyze_symptom(symptoms)
                st.write("### Diagnosis:")
                st.write(diagnosis)
        else:
            st.warning("Please enter at least one symptom.")

if app_mode == "Image Analysis":
    display_image_and_content(image_analysis_small, " ")
    
    #st.write("Upload and analyze medical images like MRI, CT, or X-ray images.")

    # File uploader for medical images
    uploaded_file = st.file_uploader("Upload a medical image (MRI, CT, X-ray, or skin images)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            # Open the uploaded image
            image = Image.open(uploaded_file)
            
            # Check if the image mode is 'RGBA' and convert to 'RGB' for JPEG format
            if image.mode == "RGBA":
                image = image.convert("RGB")
            
            # Display the image
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Prepare the image for analysis
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='JPEG')  # Convert image to JPEG format
            image_bytes.seek(0)

            # Create the image part for the model
            image_part = {
                "mime_type": "image/jpeg",
                "data": image_bytes.read()  # Use the image bytes directly
            }

            # Prepare the prompt
            prompt_parts = [image_part, system_prompt]

            # Generate a response based on the prompt and image
            with st.spinner('Analyzing the image...'):
                response = model.generate_content(prompt_parts)
                st.write(response.text)

        except OSError as e:
            # Handle cases where the image cannot be processed
            st.error(f"Error occurred: {e}. Please upload a valid medical image in PNG, JPG, or JPEG format.")

import streamlit as st
from streamlit_chat import message
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback, StreamingStdOutCallbackHandler, FileCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import openai
from loguru import logger

# Medical Chatbot Page
if app_mode == "Medical Chatbot":
    st.title("Medical Chatbot")
    display_image_and_content(chatbot_small, "Ask medical questions and get assistance from the AI-powered chatbot.")

    llm = OpenAI(openai_api_key=open_api_key)

    # Initialize session state for messages if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the input box after the image (in the main section, not sidebar)
    user_input = st.text_input("Enter your query and don't forget to hit enter:")

    # Logger setup
    logfile = "output.log"
    logger.add(logfile, colorize=True, enqueue=True)
    handler = FileCallbackHandler(logfile)
    
    # Initialize the model
    model = ChatOpenAI(
        openai_api_key=open_api_key,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0.7
    )

    # System template for chatbot with conversation history
    sys_template = """
    You are a helpful assistant that assists users with medical questions. Maintain the context of the conversation.
    Current conversation:
    {conversation}
    User's query: {query}
    """

    # Prepare conversation history to pass to the prompt
    def get_conversation_history(messages):
        history = ""
        for i, msg in enumerate(messages):
            role = "User" if i % 2 == 0 else "AI"
            history += f"{role}: {msg.content}\n"
        return history

    # Create the LLM chain
    chain1 = LLMChain(llm=model, prompt=PromptTemplate(
        input_variables=["conversation", "query"],
        template=sys_template
    ), callbacks=[handler], verbose=True)

    # Handle user input
    if user_input:
        # Add user message to session state
        st.session_state.messages.append(HumanMessage(content=user_input))

        # Get conversation history
        conversation_history = get_conversation_history(st.session_state.messages[:-1])

        # Get response from the model and append it to the session state
        with get_openai_callback() as cb:
            response = chain1.run(conversation=conversation_history, query=user_input)
        st.session_state.messages.append(AIMessage(content=response))

    # Display conversation messages
    messages = st.session_state.messages

    for i, msg in enumerate(messages):
        if i % 2 == 0:
            message(msg.content, is_user=True)
        else:
            message(msg.content, is_user=False)

# Footer
st.markdown('<div class="footer">AI Healthcare App &copy; 2024</div>', unsafe_allow_html=True)