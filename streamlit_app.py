import json
import os
import warnings
from typing import Optional

import requests
import threading
import streamlit as st
import random
import time
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
ASTRA_KEY = st.secrets["ASTRA_DB_APP_TOKEN"]
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
TAVILY_KEY = st.secrets["TAVILY_API_KEY"]

# Define your API and flow settings
BASE_API_URL = st.secrets["BASE_API_URL"]
FLOW_ID = st.secrets["FLOW_ID"] # Change new FLOW ID if the app is rebuilt
ENDPOINT = ""  # Set to empty string if not using a specific endpoint

TWEAKS = {
  "ChatInput-NCNXb": {
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
  "Prompt-F9Hul": {
    "template": "You are an expert on company's sustainability report and it circular economy efforts and impact.\nYou will get to search for this information on {company}'s sustainability report. Please be neutral and scrutinise the information.\n\n1.Efficiency and Optimization\nDefinition: Companies efforts are in replacing fossil based energy source with renewables, reducing the waste of energy and materials during manufacturing and operations.\nExample Phrases:  \"optimization in material use\", \"make the product lighter\", \"avoiding waste(production)\", \"We reduce material use\", \"We reduce water/energy use in manufacturing\", \"We run production lines using renewable energy\", \"We continuously improve processes to minimize production scraps\", \"We upgrade machinery to conserve energy\", \"We optimize logistics to reduce carbon emissions\"\n\nIdentify Targets:\nUse the example phrases as  a guide to identify their targets related. DO NOT present this example these phrases as evidences.\n\nFor each Targets,\nEvaluating the Targets:\n-Quantitative: There is a clear metric or deadline.\n-Qualitative: A stated ambition without firm timelines or numbers.\n-No pledge: No mention of any circular end target.\nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\nProvide evidence (quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular data of targets]:\n-Targets Name\n-Targets Detailed Description\n-Target Value (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factor (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
    "tool_placeholder": "",
    "company": ""
  },
  "Agent-Lc0Df": {
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
  "Prompt-RmhSW": {
    "template": "You are an expert research assistant.\n\nCreate a focused research plan that will guide our search.\n\nFormat your response exactly as:\n\nRESEARCH OBJECTIVE:\n[Clear statement of research goal]\n\nKEY SEARCH QUERIES:\n1. [Primary academic search query]\n2. [Secondary search query]\n3. [Alternative search approach]\n\nSEARCH PRIORITIES:\n- [What types of sources to focus on]\n- [Key aspects to investigate]\n- [Specific areas to explore]",
    "tool_placeholder": ""
  },
  "OpenAIModel-nWF7k": {
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
  "Prompt-MSdxo": {
    "template": "\n\nRESEARCH PLAN: {previous_response}\n\nUse Tavily Search to investigate the queries and analyse the findings\nFocus on academic and reliable sources.\n\nSteps:\n1. Search using provided queries\n2. Analyse search results\n3. Verify source credibility\n4. Extract key findings\n\nFormat findings as:\nTitle: [Company's name] Circular End Target Category 1 Efficiency and Optimisation\n\nEvaluating the Each Targets into either:\n-Quantitative: \n-Qualitative: \n-No pledge: \nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\n\nProvide evidence (url, quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular format]:\n-Targets Names\n-Targets Descriptions\n-Target Values (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factors (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
    "tool_placeholder": "",
    "previous_response": ""
  },
  "ParseData-TaQpS": {
    "sep": "\n",
    "template": "{text}"
  },
  "Prompt-BrqAt": {
    "template": "{context}\n\n---\n\nGiven the context above, answer the question as best as possible.\nHOWEVER, If there is no findings, please leave it empty and state there is no data on the database and additional web search data will complete this.\nName the Title as: [Title, vector database]\nPlease also provide the page number of the evidence. check the name of the company and the findings from this, IF the findings are not exat matchin the company name, please indicate with THE VECTOR DATABASE DOES NOT CONTAIN THE DATA and leave it empty!\n\n Question: {question}\n\nAnswer: ",
    "tool_placeholder": "",
    "context": "",
    "question": ""
  },
  "AstraDB-bH0bp": {
    "advanced_search_filter": "{}",
    "api_endpoint": "db_nov",
    "astradb_vectorstore_kwargs": "{}",
    "autodetect_collection": True,
    "collection_name": "circularity_index_2",
    "content_field": "",
    "d_api_endpoint": "https://42d07a62-a07f-42ca-b1b5-430fb4ecb099-us-east-2.apps.astra.datastax.com",
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
  "OpenAIEmbeddings-fqjnM": {
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
  "Prompt-MWKs9": {
    "template": "You are an expert on company's sustainability report and it circular economy efforts and impact.\nYou will get to search for this information on {company}'s sustainability report. Please be neutral and scrutinise the information.\n\n1.Efficiency and Optimization\nDefinition: Companies efforts are in replacing fossil based energy source with renewables, reducing the waste of energy and materials during manufacturing and operations.\nExample Phrases:  \"optimization in material use\", \"make the product lighter\", \"avoiding waste(production)\", \"We reduce material use\", \"We reduce water/energy use in manufacturing\", \"We run production lines using renewable energy\", \"We continuously improve processes to minimize production scraps\", \"We upgrade machinery to conserve energy\", \"We optimize logistics to reduce carbon emissions\"\n\nIdentify Targets:\nUse the example phrases as  a guide to identify their targets related. DO NOT present this example these phrases as evidences.\n\nFor each Targets,\nEvaluating the Targets:\n-Quantitative: There is a clear metric or deadline.\n-Qualitative: A stated ambition without firm timelines or numbers.\n-No pledge: No mention of any circular end target.\nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\nProvide evidence (quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular data of targets]:\n-Targets Name\n-Targets Detailed Description\n-Target Value (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factor (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
    "tool_placeholder": "",
    "company": ""
  },
  "OpenAIModel-0r2jf": {
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
  "Prompt-7AhIs": {
    "template": "You are an expert on circular economy and sustainability reports.\nPlease combine these two information, do not simplify it.\n\nFormat must contain a table where you can see all the relevant information.\nPlease Make sure that the title is formatted in H1, named as Target 1: Efficiency and Optimization, [company]",
    "tool_placeholder": ""
  },
  "CombineText-d1qM7": {
    "delimiter": " _____",
    "text1": "",
    "text2": ""
  },
  "CombineText-LVvsZ": {
    "delimiter": " ______________________________________________________",
    "text1": "",
    "text2": ""
  },
  "CombineText-HCE9h": {
    "delimiter": " ______________________________________________________",
    "text1": "",
    "text2": ""
  },
  "CombineText-7f2E7": {
    "delimiter": " ______________________________________________________",
    "text1": "",
    "text2": ""
  },
  "OpenAIModel-AcCtm": {
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
    "stream": True,
    "system_message": "",
    "temperature": 0.1
  },
  "ChatOutput-htbJ3": {
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
  "Prompt-oWxIf": {
    "template": "You are a professional and academic writer. \n\nContext: You are getting 5 different documents from different agents.\n\nTasks:\n1. Compare the 5 of them closely, and remove duplicated information.\n-Be aware that the emissions or anything related to sustainable fuel, must go into the target 1\n2. Rearrange the contents from Target1 to Targt5 in a sequential order.\n3. Remove activities from the original data and compile as Appendix: Please take a look, since these are qualitative and need verification.\n\nSuccess Criteria:\nMaintain the length and the details from the input text\nMaintain the source and citation for the evidences\nTabular summary generation\nClearly divided Target1, Target2, Target3. Target4, Target5, Overall conclusion, limitation, appendix.\nOverall evaluation based on the compiled data.\n\nBe Aware:\nAnything related to emission reduction or fuel usage must go to Target1 bucket and removed from others if they are duplicated.\n",
    "tool_placeholder": ""
  },
  "TavilySearchComponent-uhLP4": {
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
  "Agent-vvIHZ": {
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
  },
  "ChatOutput-yoOv1": {
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
  "ChatOutput-dZeJP": {
    "background_color": "",
    "chat_icon": "",
    "data_template": "{text}",
    "input_value": "",
    "sender": "Machine",
    "sender_name": "AI",
    "session_id": "",
    "should_store_message": True,
    "text_color": ""
  }
}

##########################################
# Flow-Related Functions
##########################################
def run_flow(message: str,
             endpoint: str,
             output_type: str = "chat",
             input_type: str = "chat",
             tweaks: Optional[dict] = None,
             api_key: Optional[str] = None) -> dict:
    """
    Synchronous call to your Hugging Face Space API.
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

##########################################
# Deduped Agent Steps
##########################################
def find_unique_agent_steps(data, seen_blocks=None):
    """
    Recursively search the JSON for 'content_blocks' with title='Agent Steps'.
    Deduplicate them by a signature so we only get each unique block once.
    """
    if seen_blocks is None:
        seen_blocks = set()

    agent_steps_list = []

    if isinstance(data, dict):
        content_blocks = data.get('content_blocks')
        if isinstance(content_blocks, list):
            for block in content_blocks:
                if block.get('title') == "Agent Steps":
                    contents = block.get('contents', [])
                    signature = make_block_signature(contents)
                    if signature not in seen_blocks:
                        seen_blocks.add(signature)
                        agent_steps_list.append(contents)

        for val in data.values():
            agent_steps_list.extend(find_unique_agent_steps(val, seen_blocks=seen_blocks))

    elif isinstance(data, list):
        for element in data:
            agent_steps_list.extend(find_unique_agent_steps(element, seen_blocks=seen_blocks))

    return agent_steps_list

def make_block_signature(contents):
    """
    Combine all 'text' fields of each step item into one string
    to detect duplicates.
    """
    lines = []
    for step in contents:
        txt = step.get('text', '').strip()
        if txt:
            lines.append(txt)
    # Return a single string signature
    return "\n".join(lines)

##########################################
# JSON Display
##########################################
def st_stream_json_output(json_data):
    """
    1) Show raw JSON
    2) Then recursively search 'text' fields
    """
    st.subheader("Raw JSON Response")
    st.json(json_data)

    st.subheader("Text Fields Found")
    def recursive_parse(data):
        if isinstance(data, dict):
            for k, v in data.items():
                if k == "text" and isinstance(v, str):
                    st.write(f"• {v}")
                else:
                    recursive_parse(v)
        elif isinstance(data, list):
            for item in data:
                recursive_parse(item)

    recursive_parse(json_data)

##########################################
# STREAMLIT UI
##########################################
ICON_BLUE = ".static/Logo-Blue-Indeed.png"
ICON_WHITE = ".static/Logo-White-Indeed.png"

st.logo(ICON_BLUE, icon_image=ICON_WHITE, size="large", link="https://www.indeed-innovation.com/")
st.set_page_config(page_title="INDEED Circularity Index", page_icon="🌍", layout="wide")

def icon(emoji: str):
    st.write(f'<span style="font-size: 78px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)

icon("Circularity Index Researcher")

st.subheader(
    "Ten specialized agents collaborate with you to analyze and interpret complex sustainability reports packed with data.",
    divider="rainbow",
    anchor=False
)

# Sidebar styling
with st.sidebar:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #DFE3F2;
        }
        [data-testid="stSidebar"] * {
            color: #0D1327 !important;
        }
        [data-testid="stTextInput"] input {
            color: white !important;
        }
        [data-testid="stForm"] {
            background-color: #E3FFCC !important;
            border: 1px solid #FFFFFF !important;
            border-radius: 10px !important;
            padding: 15px !important;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.5) !important;
        }
        [data-testid="stTextInput"] input::placeholder {
            color: white !important;
        }
        [data-testid="stFormSubmitButton"] button {
            background-color: #FFFFF !important;
            border: 1px solid #0D1327 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.header("👇 Enter details")
    with st.form("my_form"):
        company = st.text_input("Which company are you looking for?", placeholder="input company name")
        submitted = st.form_submit_button("🔍Search")


##########################################
# BACKGROUND THREAD + FAKE LOAD
##########################################
if submitted and company:
    # This container holds the final JSON result from the thread
    result_container = {"data": None}

    def background_run_flow():
        TWEAKS["ChatInput-NCNXb"]["input_value"] = company
        flow_result = run_flow(
            message=company,
            endpoint=ENDPOINT or FLOW_ID,
            tweaks=TWEAKS,
            api_key=API_KEY
        )
        result_container["data"] = flow_result

    # Start the thread
    worker_thread = threading.Thread(target=background_run_flow)
    worker_thread.start()

    # Show "fake" steps while the request is running
    steps_sequence = [
        ("initializing...", 4, 6),
        ("gathering agents...", 4, 6),
        ("assigning tasks...", 5, 8),
        ("running agent flow...", 4, 4),
        ("running agent flow...", 4, 4),
        ("running agent flow...", 4, 4),
        ("checking information...", 4, 4),
        ("checking information...", 4, 4),
        ("checking information...", 4, 4),
        ("running agent flow...", 4, 4),
        ("expanding searches...", 4, 4),
        ("expanding searches...", 4, 4),
        ("running agent flow...", 8, 15),
    ]
    for step_message, min_sec, max_sec in steps_sequence:
        if worker_thread.is_alive():
            st.toast(step_message, icon=":material/hourglass_empty:")
            time.sleep(random.randint(min_sec, max_sec))
        else:
            break

    # Ensure the thread is done
    worker_thread.join()

    # Retrieve the JSON result
    result = result_container["data"]

    # ---------------------------------------------------------
    # 1) Identify the final output chunk
    #    We'll assume it's the last item in result["outputs"][0]["outputs"]
    # ---------------------------------------------------------
    outputs_list = result["outputs"][0]["outputs"]
    # Make sure we have enough items
    if len(outputs_list) < 3:
        st.write("Not enough outputs to show final vs. agents.")
        # Optional: just show entire JSON and return
        st_stream_json_output(result)
        st.stop()

    # final chunk is the last one
    final_chunk = outputs_list[-3]  # chunk [-3] --> 0 if indexing 0..1..2
    # agent1 chunk is the second-to-last
    agent1_chunk = outputs_list[-2] 
    # agent2 chunk is the third-to-last
    agent2_chunk = outputs_list[-1]



    # ---------------------------------------------------------
    # 2) Show the final output FIRST
    # ---------------------------------------------------------
    st.subheader("Final Output")
    try: #result["outputs"][0]["outputs"][0]["results"]["message"]["data"]
        final_output_text = final_chunk["results"]["message"]["text"]
        st.markdown(final_output_text)
    except (KeyError, IndexError) as e:
        st.write(f"Error parsing final output: {e}")

    # ---------------------------------------------------------
    # 3) Agents in Expanders
    #    We'll call them "Agent1: AI Web Search" and "Agent2: Document Search"
    #    We'll show the input + output from each chunk
    # ---------------------------------------------------------
    st.write("---")    
    st.subheader("Agents Info")

    # -- Agent1
    with st.expander("Agent1: AI Web Search", expanded=False):
        try:
            # Show the input
            # Usually it's in agent1_chunk["results"]["message"] or agent1_chunk["logs"], 
            # or sometimes "content_blocks"
            # We'll do a guess:
            input_for_agent1 = agent1_chunk["results"]["message"]["content_blocks"][0]["contents"][0]["text"]

            # If you have a known path, do that. Example:
            # input_for_agent1 = agent1_chunk["results"]["message"]["data"]["content_blocks"][0]["contents"][0]["text"]
            # But that depends on your actual JSON structure.
            
            # Show the output
            agent1_output_text = agent1_chunk["results"]["message"]["text"]
            st.write("### Input\n", input_for_agent1)
            st.write("### Output\n", agent1_output_text)
        except (KeyError, IndexError) as e:
            st.write(f"Error parsing Agent1 chunk: {e}")

    # -- Agent2
    with st.expander("Agent2: Document Search", expanded=False):
        try:
            #input_for_agent2 = agent2_chunk["results"]["message"]["content_blocks"][0]["contents"][0]["text"]
            # same approach
            st.markdown("Vector Database Search Results:")
            agent2_output_text = agent2_chunk["results"]["message"]["text"]
            #st.write("### Input\n", input_for_agent2)
            st.write("### Output\n", agent2_output_text)
        except (KeyError, IndexError) as e:
            st.write(f"Error parsing Agent2 chunk: {e}")

    # ---------------------------------------------------------
    # 4) Show entire JSON if you want
    # ---------------------------------------------------------
    st.write("---")
    #st_stream_json_output(result)