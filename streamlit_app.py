import json
import os
import warnings
from typing import Optional

import requests
import streamlit as st
from dotenv import load_dotenv

# Optionally import langflow's upload function
try:
    from langflow.load import upload_file
except ImportError:
    warnings.warn("Langflow provides a function to help you upload files to the flow. Please install langflow to use it.")
    upload_file = None

# Load environment variables
#load_dotenv()

# I need to have this to make an API call via Hugginface Space
API_KEY = st.secrets["HF_API_KEY"] #os.getenv("HF_API_KEY") 

# Define your API and flow settings
BASE_API_URL = "https://jeo-jang-langflownew.hf.space"
FLOW_ID = FLOW_ID = "c8940c66-6184-45e0-ad70-d4ec8b64eccf" #"ea36bda2-6cde-4d57-be90-59d2688dd090" was lost due to rebuild app.
ENDPOINT = ""  # Set to empty string if not using a specific endpoint

# Define your tweaks dictionary. Notice that we set the user input to an empty string.
# Later, we update it with the value from the text input. Please see below with TWEAKS[blah blah][blah blah]
TWEAKS = {
    "ChatInput-8zFTw": {
        "background_color": "",
        "chat_icon": "",
        "files": "",
        "input_value": "",  # This will be updated with the user's input
        "sender": "User",
        "sender_name": "User",
        "session_id": "",
        "should_store_message": True,
        "text_color": ""
    },
    "TextInput-Q7ye5": {
        "input_value": "- Thread must be 5-7 tweets long - Each tweet should be self-contained but flow naturally to the next - Include relevant technical details while keeping language accessible - Use emojis sparingly but effectively - Include a clear call-to-action in the final tweet - Highlight key benefits and innovative aspects - Maintain professional but engaging tone"
    },
    "ChatOutput-wTQTP": {
        "background_color": "",
        "chat_icon": "",
        "data_template": "{text}",
        "input_value": "",
        "sender": "Machine",
        "sender_name": "AI",
        "session_id": "",
        "should_store_message": True,
        "text_color": ""
    },
    "TextInput-BQHnI": {
        "input_value": "thread"
    },
    "TextInput-X18Sl": {
        "input_value": "English"
    },
    "TextInput-VJ8Zw": {
        "input_value": "- Tech startup focused on Vegan food"
    },
    "TextInput-DnWUD": {
        "input_value": "- Professional yet approachable - annoyed and negative"
    },
    "TextInput-8FJGR": {
        "input_value": "Vegan product Company"
    },
    "Prompt-rZryL": {
        "CONTENT_GUIDELINES": "",
        "OUTPUT_FORMAT": "",
        "OUTPUT_LANGUAGE": "",
        "PROFILE_DETAILS": "",
        "PROFILE_TYPE": "",
        "TONE_AND_STYLE": "",
        "template": (
            "<Instructions Structure>\nIntroduce the task of generating tweets or tweet threads based on the provided inputs\n\n"
            "Explain each input variable:\n\n{{PROFILE_TYPE}}\n\n{{PROFILE_DETAILS}}\n\n{{CONTENT_GUIDELINES}}\n\n{{TONE_AND_STYLE}}\n\n"
            "{{CONTEXT}}\n\n{{OUTPUT_FORMAT}}\n\n{{OUTPUT_LANGUAGE}}\n\n"
            "Provide step-by-step instructions on how to analyze the inputs to determine if a single tweet or thread is appropriate\n\n"
            "Give guidance on generating tweet content that aligns with the profile, guidelines, tone, style, and context\n\n"
            "Explain how to format the output based on the {{OUTPUT_FORMAT}} value\n\n"
            "Provide tips for creating engaging, coherent tweet content\n\n</Instructions Structure>\n\n"
            "<Instructions>\nYou are an AI tweet generator that can create standalone tweets or multi-tweet threads based on a variety of inputs about the desired content. Here are the key inputs you will use to generate the tweet(s):\n\n"
            "<profile_type>\n\n{PROFILE_TYPE}\n\n</profile_type>\n\n"
            "<profile_details>\n\n{PROFILE_DETAILS}\n\n</profile_details>\n\n"
            "<content_guidelines>\n\n{CONTENT_GUIDELINES}\n\n</content_guidelines>\n\n"
            "<tone_and_style>\n\n{TONE_AND_STYLE}\n\n</tone_and_style>\n\n"
            "<output_format>\n\n{OUTPUT_FORMAT}\n\n</output_format>\n\n"
            "<output_language>\n\n{OUTPUT_LANGUAGE}\n\n</output_language>\n\n"
            "To generate the appropriate tweet(s), follow these steps:\n\n"
            "<output_determination>\n\nCarefully analyze the {{PROFILE_TYPE}}, {{PROFILE_DETAILS}}, {{CONTENT_GUIDELINES}}, {{TONE_AND_STYLE}}, and {{CONTEXT}} to determine the depth and breadth of content needed.\n\n"
            "If the {{OUTPUT_FORMAT}} is \"single_tweet\", plan to convey the key information in a concise, standalone tweet.\n\n"
            "If the {{OUTPUT_FORMAT}} is \"thread\" or if the content seems too complex for a single tweet, outline a series of connected tweets that flow together to cover the topic.\n\n"
            "</output_determination>\n\n"
            "<content_generation>\n\nBrainstorm tweet content that aligns with the {{PROFILE_TYPE}} and {{PROFILE_DETAILS}}, adheres to the {{CONTENT_GUIDELINES}}, matches the {{TONE_AND_STYLE}}, and incorporates the {{CONTEXT}}.\n\n"
            "For a single tweet, craft the most engaging, informative message possible within the 280 character limit.\n\n"
            "For a thread, break down the content into distinct yet connected tweet-sized chunks. Ensure each tweet flows logically into the next to maintain reader engagement. Use transitional phrases as needed to link tweets.\n\n"
            "</content_generation>\n\n"
            "<formatting>\nFormat the output based on the {{OUTPUT_FORMAT}}:\n\n"
            "For a single tweet, provide the content.\n\n"
            "For a thread, include each tweet inside numbered markdown list.\n\n"
            "</formatting>\n\n"
            "<tips>\nFocus on creating original, engaging content that provides value to the intended audience.\n\n"
            "Optimize the tweet(s) for the 280 character limit. Be concise yet impactful.\n\n"
            "Maintain a consistent voice that matches the {{TONE_AND_STYLE}} throughout the tweet(s).\n\n"
            "Include calls-to-action or questions to drive engagement when appropriate.\n\n"
            "Double check that the final output aligns with the {{PROFILE_DETAILS}} and {{CONTENT_GUIDELINES}}.\n\n"
            "</tips>\n\nNow create a Tweet or Twitter Thread for this context:\n\n"
        ),
        "tool_placeholder": ""
    },
    "OpenAIModel-W0BHu": {
        "api_key": {
            "load_from_db": False,
            "value": st.secrets["OPENAI_API_KEY"] #os.getenv("OPENAI_API_KEY")
        },
        "input_value": "",
        "json_mode": False,
        "max_tokens": None,
        "model_kwargs": {},
        "model_name": "gpt-4o-mini",
        "openai_api_base": "",
        "seed": 1,
        "stream": False,
        "system_message": "",
        "temperature": 0.1
    }
}

def run_flow(message: str,
             endpoint: str,
             output_type: str = "chat",
             input_type: str = "chat",
             tweaks: Optional[dict] = None,
             api_key: Optional[str] = None) -> dict:
    """
    Run a flow with the given parameters.
    """
    api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"
    payload = {
        "output_type": output_type,
        "input_type": input_type,
    }
    if tweaks:
        payload["tweaks"] = tweaks
    headers = {"x-api-key": api_key} if api_key else None
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

# ------------------ Streamlit App UI ------------------

st.title("Langflow Streamlit Tester")

# Create a text input box for the user's message
user_message = st.text_input("Enter your message:")

# Create text input selection and boxes for the pormpts
language_selection = st.selectbox(
    "Which language should the agent use?",
    ("English", "German", "French"),
    placeholder="Select post language...",
)

tone_selection = st.text_input("Enter the tone of the post:")


# If you want to allow file upload as well, you can use st.file_uploader (optional)
# uploaded_file = st.file_uploader("Upload a file", type=["txt", "csv", "json"])

if st.button("Submit"):
    if user_message:
        # Update the tweak for the chat input with the user's message
        TWEAKS["ChatInput-8zFTw"]["input_value"] = user_message
        TWEAKS["TextInput-X18Sl"]["input_value"] = language_selection
        TWEAKS["TextInput-DnWUD"]["input_value"] = tone_selection

        # Optionally, if you support file uploads via langflow's upload_file function,
        # you can add logic here to update the tweaks accordingly.
        # For example:
        # if uploaded_file and upload_file:
        #     # Save the file temporarily and call upload_file
        #     with open("temp_file", "wb") as f:
        #         f.write(uploaded_file.getbuffer())
        #     TWEAKS = upload_file(
        #         file_path="temp_file",
        #         host=BASE_API_URL,
        #         flow_id=ENDPOINT or FLOW_ID,
        #         components=["<component_name>"],
        #         tweaks=TWEAKS
        #     )

        # Run the flow with the user input and updated tweaks
        result = run_flow(
            message=user_message,
            endpoint=ENDPOINT or FLOW_ID,
            tweaks=TWEAKS,
            api_key=API_KEY
        )

        # Extract the tweet text from the nested JSON structure.
        # The exact path depends on the structure of your response.
        # Based on the response you provided, one way might be:
        try:
            tweet_text = result["outputs"][0]["outputs"][0]["results"]["message"]["text"]
        except (KeyError, IndexError) as e:
            tweet_text = "Error parsing tweet text from response: " + str(e)

        # Now display the tweet text using st.write or st.markdown
        st.subheader("Tweet Output")
        st.markdown(tweet_text)
