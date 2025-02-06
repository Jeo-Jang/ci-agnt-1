import json
import os
import time
import warnings
import logging
from typing import Optional

import httpx
import requests
import streamlit as st
from dotenv import load_dotenv

# Optionally import langflow's upload function
try:
    from langflow.load import upload_file
except ImportError:
    warnings.warn(
        "Langflow provides a function to help you upload files to the flow. Please install langflow to use it."
    )
    upload_file = None

# Optionally load environment variables
# load_dotenv()

# API keys (retrieved from st.secrets)
API_KEY = st.secrets["HF_API_KEY"]  # or os.getenv("HF_API_KEY")
ASTRA_KEY = st.secrets["ASTRA_DB_APP_TOKEN"]
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
TAVILY_KEY = st.secrets["TAVILY_API_KEY"]

# API and flow settings
BASE_API_URL = "https://jeo-jang-langflownew.hf.space"  # Use your deployed host here
FLOW_ID = "958fd1d6-d2d1-4fee-98bf-235fccb0b89b"  # Change if the app is rebuilt
ENDPOINT = ""  # Set to empty string if not using a specific endpoint

TWEAKS = {
  "ChatInput-PKiJr": {
    "files": "",
    "background_color": "",
    "chat_icon": "",
    "input_value": "", # This will be input from the user
    "sender": "User",
    "sender_name": "User",
    "session_id": "",
    "should_store_message": True,
    "text_color": ""
  },
  "Prompt-WAIhH": {
    "template": "You are an expert on company's sustainability report and it circular economy efforts and impact.\nYou will get to search for this information on {company}'s sustainability report. Please be neutral and scrutinise the information.\n\n1.Efficiency and Optimization\nDefinition: Companies efforts are in replacing fossil based energy source with renewables, reducing the waste of energy and materials during manufacturing and operations.\nExample Phrases:  \"optimization in material use\", \"make the product lighter\", \"avoiding waste(production)\", \"We reduce material use\", \"We reduce water/energy use in manufacturing\", \"We run production lines using renewable energy\", \"We continuously improve processes to minimize production scraps\", \"We upgrade machinery to conserve energy\", \"We optimize logistics to reduce carbon emissions\"\n\nIdentify Targets:\nUse the example phrases as  a guide to identify their targets related. DO NOT present this example these phrases as evidences.\n\nFor each Targets,\nEvaluating the Targets:\n-Quantitative: There is a clear metric or deadline.\n-Qualitative: A stated ambition without firm timelines or numbers.\n-No pledge: No mention of any circular end target.\nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\nProvide evidence (quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular data of targets]:\n-Targets Name\n-Targets Detailed Description\n-Target Value (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factor (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
    "tool_placeholder": "",
    "company": ""
  },
  "Agent-dgYWE": {
    "add_current_date_tool": True,
    "agent_description": "A helpful assistant with access to the following tools:",
    "agent_llm": "OpenAI",
    "api_key": {
        "load_from_db": False,
        "value": OPENAI_KEY
        },
    "handle_parsing_errors": True,
    "input_value": "",
    "json_mode": False,
    "max_iterations": 15,
    "max_tokens": None,
    "model_kwargs": {},
    "model_name": "gpt-4o",
    "n_messages": 100,
    "openai_api_base": "",
    "order": "Ascending",
    "seed": 1,
    "sender": "Machine and User",
    "sender_name": "",
    "session_id": "",
    "system_prompt": "You are a research analyst with access to Tavily AI Search",
    "temperature": 0.1,
    "template": "{sender_name}: {text}",
    "verbose": True
  },
  "Prompt-meXq8": {
    "template": "You are an expert research assistant.\n\nCreate a focused research plan that will guide our search.\n\nFormat your response exactly as:\n\nRESEARCH OBJECTIVE:\n[Clear statement of research goal]\n\nKEY SEARCH QUERIES:\n1. [Primary academic search query]\n2. [Secondary search query]\n3. [Alternative search approach]\n\nSEARCH PRIORITIES:\n- [What types of sources to focus on]\n- [Key aspects to investigate]\n- [Specific areas to explore]",
    "tool_placeholder": ""
  },
  "OpenAIModel-EkPZv": {
    "api_key": {
        "load_from_db": False,
        "value": OPENAI_KEY #os.getenv("OPENAI_API_KEY")
    },
    "input_value": "",
    "json_mode": False,
    "max_tokens": None,
    "model_kwargs": {},
    "model_name": "gpt-4o",
    "openai_api_base": "",
    "seed": 1,
    "stream": False, # what happens if it's True?
    "system_message": "",
    "temperature": 0.1
  },
  "Prompt-coXDM": {
    "template": "\n\nRESEARCH PLAN: {previous_response}\n\nUse Tavily Search to investigate the queries and analyse the findings\nFocus on academic and reliable sources.\n\nSteps:\n1. Search using provided queries\n2. Analyse search results\n3. Verify source credibility\n4. Extract key findings\n\nFormat findings as:\nTitle: [Company's name] Circular End Target Category 1 Efficiency and Optimisation\n\nEvaluating the Each Targets into either:\n-Quantitative: \n-Qualitative: \n-No pledge: \nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\n\nProvide evidence (quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular format]:\n-Targets Names\n-Targets Descriptions\n-Target Values (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factors (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
    "tool_placeholder": "",
    "previous_response": ""
  },
  "ParseData-bUyDz": {
    "sep": "\n",
    "template": "{text}"
  },
  "Prompt-OPWHS": {
    "template": "{context}\n\n---\n\nGiven the context above, answer the question as best as possible.\nHOWEVER, If there is no findings, please leave it empty and state there is no data on the database and additional web search data will complete this.\nName the Title as: [Title, vector database]\nPlease also provide the page number of the evidence.\n\nQuestion: {question}\n\nAnswer: ",
    "tool_placeholder": "",
    "context": "",
    "question": ""
  },
  "AstraDB-tQShy": {
    "advanced_search_filter": "{}",
    "api_endpoint": "db_nov",
    "astradb_vectorstore_kwargs": "{}",
    "autodetect_collection": True,
    "collection_name": "circularity_index_2",
    "content_field": "",
    #"d_api_endpoint": "https://42d07a62-a07f-42ca-b1b5-430fb4ecb099-us-east-2.apps.astra.datastax.com",
    "deletion_field": "",
    "embedding_choice": "Embedding Model",
    "environment": "",
    "ignore_invalid_documents": False,
    "keyspace": "keyspace001",
    "number_of_results": 4,
    "search_query": "",
    "search_score_threshold": 0,
    "search_type": "Similarity",
    "token": {
        "load_from_db": False,
        "value": ASTRA_KEY,
    }
  },
  "OpenAIEmbeddings-TrLB6": {
    "chunk_size": 1000,
    "client": "",
    "default_headers": {},
    "default_query": {},
    "deployment": "",
    "dimensions": None,
    "embedding_ctx_length": 1536,
    "max_retries": 3,
    "model": "text-embedding-3-small",
    "model_kwargs": {},
    "openai_api_base": "",
    "openai_api_key": {
        "load_from_db": False,
        "value": OPENAI_KEY
        },
    "openai_api_type": "",
    "openai_api_version": "",
    "openai_organization": "",
    "openai_proxy": "",
    "request_timeout": None,
    "show_progress_bar": False,
    "skip_empty": False,
    "tiktoken_enable": True,
    "tiktoken_model_name": ""
  },
  "Prompt-9VKl4": {
    "template": "You are an expert on company's sustainability report and it circular economy efforts and impact.\nYou will get to search for this information on {company}'s sustainability report. Please be neutral and scrutinise the information.\n\n1.Efficiency and Optimization\nDefinition: Companies efforts are in replacing fossil based energy source with renewables, reducing the waste of energy and materials during manufacturing and operations.\nExample Phrases:  \"optimization in material use\", \"make the product lighter\", \"avoiding waste(production)\", \"We reduce material use\", \"We reduce water/energy use in manufacturing\", \"We run production lines using renewable energy\", \"We continuously improve processes to minimize production scraps\", \"We upgrade machinery to conserve energy\", \"We optimize logistics to reduce carbon emissions\"\n\nIdentify Targets:\nUse the example phrases as  a guide to identify their targets related. DO NOT present this example these phrases as evidences.\n\nFor each Targets,\nEvaluating the Targets:\n-Quantitative: There is a clear metric or deadline.\n-Qualitative: A stated ambition without firm timelines or numbers.\n-No pledge: No mention of any circular end target.\nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\nProvide evidence (quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular data of targets]:\n-Targets Name\n-Targets Detailed Description\n-Target Value (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factor (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
    "tool_placeholder": "",
    "company": ""
  },
  "OpenAIModel-2cpDG": {
    "api_key": {
        "load_from_db": False,
        "value": OPENAI_KEY #os.getenv("OPENAI_API_KEY")
    },
    "input_value": "",
    "json_mode": False,
    "max_tokens": None,
    "model_kwargs": {},
    "model_name": "gpt-4o",
    "openai_api_base": "",
    "seed": 1,
    "stream": False,
    "system_message": "",
    "temperature": 0.1
  },
  "Prompt-zwFjX": {
    "template": "You are an expert on circular economy and sustainability reports.\nPlease combine these two information, do not simplify it.\n\nFormat must contain a table where you can see all the relevant information.\nPlease Make sure that the title is formatted in H1, named as Target 1: Efficiency and Optimization, [company]",
    "tool_placeholder": ""
  },
  "CombineText-3kUiI": {
    "delimiter": " _____",
    "text1": "",
    "text2": ""
  },
  "CombineText-aiR0t": {
    "delimiter": " ______________________________________________________",
    "text1": "",
    "text2": ""
  },
  "CombineText-Pmnmp": {
    "delimiter": " ______________________________________________________",
    "text1": "",
    "text2": ""
  },
  "CombineText-ao3oD": {
    "delimiter": " ______________________________________________________",
    "text1": "",
    "text2": ""
  },
  "OpenAIModel-dyUI5": {
    "api_key": {
        "load_from_db": False,
        "value": OPENAI_KEY #os.getenv("OPENAI_API_KEY")
        },
    "input_value": "",
    "json_mode": False,
    "max_tokens": 0,
    "model_kwargs": {},
    "model_name": "gpt-4o",
    "openai_api_base": "",
    "seed": 1,
    "stream": True, # What if this is True?
    "system_message": "",
    "temperature": 0.1
  },
  "ChatOutput-dLsMn": {
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
  "Prompt-6t2ji": {
    "template": "You are a professional and academic writer. \n\nContext: You are getting 5 different documents from different agents.\n\nTasks:\n1. Compare the 5 of them closely, and remove duplicated information.\n-Be aware that the emissions or anything related to sustainable fuel, must go into the target 1\n2. Rearrange the contents from Target1 to Targt5 in a sequential order.\n3. Remove activities from the original data and compile as Appendix: Please take a look, since these are qualitative and need verification.\n\nSuccess Criteria:\nMaintain the length and the details from the input text\nMaintain the source and citation for the evidences\nTabular summary generation\nClearly divided Target1, Target2, Target3. Target4, Target5, Overall conclusion, limitation, appendix.\nOverall evaluation based on the compiled data.\n\nBe Aware:\nAnything related to emission reduction or fuel usage must go to Target1 bucket and removed from others if they are duplicated.\n",
    "tool_placeholder": ""
  },
  "TavilySearchComponent-DWGdg": {
    "api_key": {
        "load_from_db": False, 
        "value": TAVILY_KEY 
        },
    "include_answer": True,
    "include_images": True,
    "max_results": 1,
    "query": "",
    "search_depth": "advanced",
    "topic": "general",
    "tools_metadata": [
      {
        "name": "TavilySearchComponent-fetch_content",
        "description": "fetch_content(api_key: Message) - **Tavily AI** is a search engine optimized for LLMs and RAG,         aimed at efficient, quick, and persistent search results.",
        "tags": [
          "TavilySearchComponent-fetch_content"
        ]
      },
      {
        "name": "TavilySearchComponent-fetch_content_text",
        "description": "fetch_content_text(api_key: Message) - **Tavily AI** is a search engine optimized for LLMs and RAG,         aimed at efficient, quick, and persistent search results.",
        "tags": [
          "TavilySearchComponent-fetch_content_text"
        ]
      }
    ]
  },
  "Agent-x86zf": {
    "add_current_date_tool": True,
    "agent_description": "A helpful assistant with access to the following tools:",
    "agent_llm": "OpenAI",
    "api_key": {
        "load_from_db": False,
        "value": OPENAI_KEY
        },
    "handle_parsing_errors": True,
    "input_value": "",
    "json_mode": False,
    "max_iterations": 15,
    "max_tokens": None,
    "model_kwargs": {},
    "model_name": "gpt-4o",
    "n_messages": 100,
    "openai_api_base": "",
    "order": "Ascending",
    "seed": 1,
    "sender": "Machine and User",
    "sender_name": "",
    "session_id": "",
    "system_prompt": "You are a research analyst with access to Vector database",
    "temperature": 0.1,
    "template": "{sender_name}: {text}",
    "verbose": True
  }
}

# -------------------------
# Non-streaming function (from your original code)
# -------------------------
def run_flow(message: str,
             endpoint: str,
             output_type: str = "chat",
             input_type: str = "chat",
             tweaks: Optional[dict] = None,
             api_key: Optional[str] = None) -> dict:
    """
    Run a flow with a given message and optional tweaks.
    (This is similar to the original argparse-based code.)
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
    response.raise_for_status()
    return response.json()

# -------------------------
# Streaming functions (using the original snippet’s payload)
# -------------------------
def is_healthy(api_key: Optional[str] = None) -> bool:
    """
    Check if the API is healthy.
    """
    try:
        url = f"{BASE_API_URL}/health"
        headers = {"Accept": "application/json"}
        if api_key:
            headers["x-api-key"] = api_key
        response = httpx.get(url, headers=headers)
        return response.status_code == 200
    except Exception as e:
        logging.error(f"Health check error: {str(e)}")
        return False

def initiate_session(flow_id: str, input_value: str, stream: bool = True, api_key: Optional[str] = None):
    """
    Initiate a session for streaming.
    
    This version sends a payload of the form:
       {"input_value": <input_value>}
    which is how the original snippet sends it.
    """
    # Debug: Log the input_value we're sending.
    logging.info(f"Preparing to send input_value: '{input_value}'")
    if not input_value:
        logging.warning("Input value is empty! Using default value from tweaks.")
        # Optionally, use the default from tweaks if desired:
        input_value = TWEAKS["ChatInput-PKiJr"].get("input_value", "")
    
    url = f"{BASE_API_URL}/api/v1/run/{flow_id}?stream={str(stream).lower()}"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    data = {"input_value": input_value}
    logging.info(f"Initiating session with data: {data}")
    healthy = is_healthy(api_key)
    while not healthy:
        time.sleep(1)
        healthy = is_healthy(api_key)
    logging.info("Health check passed")
    
    response = httpx.post(url, json=data, headers=headers)
    response.raise_for_status()
    logging.info(f"Raw response text: {response.text}")
    if response.status_code == 200:
        try:
            return response.json()
        except Exception as e:
            logging.error("Failed to decode JSON. Response text was:")
            logging.error(response.text)
            raise e
    else:
        raise Exception("Failed to initiate session")

def stream_response(stream_url: str, session_id: str, placeholder, api_key: Optional[str] = None) -> str:
    """
    Stream output from the provided stream_url and session_id.
    Accumulates lines and updates a Streamlit placeholder.
    """
    if stream_url.startswith("/"):
        full_stream_url = f"{BASE_API_URL}{stream_url}"
    else:
        full_stream_url = stream_url
    params = {"session_id": session_id}
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    stream_output = ""
    with httpx.stream("GET", full_stream_url, params=params, headers=headers, timeout=None) as response:
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                logging.info(f"Line: {decoded_line}")
                stream_output += decoded_line
                placeholder.markdown(stream_output)
    return stream_output

# -------------------------
# Streamlit App UI
# -------------------------
st.logo(".static/Logo-Blue-Indeed.png", size="large", link="https://www.indeed-innovation.com/")
st.set_page_config(page_icon="🌍", layout="wide")

def icon(emoji: str):
    """Display a large emoji as a page icon."""
    st.write(f'<span style="font-size: 78px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)

icon("Circularity Index Researcher")

st.subheader(
    "Ten specialized agents collaborate with you to analyze and interpret complex sustainability reports packed with data.",
    divider="rainbow",
    anchor=False
)

with st.sidebar:
    st.markdown(
        """
        <style>
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #DFE3F2;
        }
        [data-testid="stSidebar"] * {
            color: #0D1327 !important;
        }
        [data-testid="stTextInput"] input {
            color: white !important;
        }
        [data-testid="stDateInput"] input {
            color: white !important;
        }
        [data-testid="stTextInput"] input::placeholder {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.header("👇 Enter details")
    with st.form("my_form"):
        company = st.text_input("Which company are you looking for?", placeholder="Input company name")
        use_streaming = st.checkbox("Stream response", value=True)
        submitted = st.form_submit_button("Submit")

if submitted:
    if company:
        if use_streaming:
            try:
                # For streaming, we send a simple payload with just "input_value"
                init_response = initiate_session(FLOW_ID, company, stream=True, api_key=API_KEY)
                logging.info(f"Init Response: {init_response}")
                session_id = init_response.get("session_id")
                artifacts = (init_response.get("outputs", [{}])[0]
                             .get("outputs", [{}])[0]
                             .get("artifacts", {}))
                if "stream_url" not in artifacts:
                    raise Exception("No stream URL returned")
                stream_url = artifacts["stream_url"]

                st.write(f"**Session ID:** {session_id}")
                st.info("Streaming response…")
                placeholder = st.empty()
                final_output = stream_response(stream_url, session_id, placeholder, api_key=API_KEY)
                st.subheader("Final Output")
                st.markdown(final_output)
            except Exception as e:
                st.error(f"Error during streaming: {e}")
                st.info("Falling back to non-streaming mode.")
                try:
                    result = run_flow(company, ENDPOINT or FLOW_ID, tweaks=TWEAKS, api_key=API_KEY)
                    try:
                        tweet_text = result["outputs"][0]["outputs"][0]["results"]["message"]["text"]
                    except (KeyError, IndexError) as err:
                        tweet_text = f"Error parsing text: {err}"
                    st.subheader("Output")
                    st.markdown(tweet_text)
                except Exception as ne:
                    st.error(f"Non-streaming call failed: {ne}")
        else:
            try:
                result = run_flow(company, ENDPOINT or FLOW_ID, tweaks=TWEAKS, api_key=API_KEY)
                try:
                    tweet_text = result["outputs"][0]["outputs"][0]["results"]["message"]["text"]
                except (KeyError, IndexError) as err:
                    tweet_text = f"Error parsing text: {err}"
                st.subheader("Output")
                st.markdown(tweet_text)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.error("Please enter a company name.")