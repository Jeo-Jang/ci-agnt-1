import json
import os
import warnings
from typing import Optional
import requests
import streamlit as st

# For optional file-upload to Langflow:
try:
    from langflow.load import upload_file
except ImportError:
    warnings.warn("Please install langflow if you want to use upload_file.")
    upload_file = None

############################
# 1) CONFIG & TWEAKS SETUP
############################
BASE_API_URL = "https://jeo-jang-langflownew.hf.space"
FLOW_ID = "958fd1d6-d2d1-4fee-98bf-235fccb0b89b" 
ENDPOINT = ""  # or a named endpoint, if you have one

# Retrieve secrets (Streamlit)
API_KEY = st.secrets["HF_API_KEY"]
ASTRA_KEY = st.secrets["ASTRA_DB_APP_TOKEN"]
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
TAVILY_KEY = st.secrets["TAVILY_API_KEY"]

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
    "stream": True,
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
  },
  "ChatOutput-928EC": {
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
  "ChatOutput-WoNda": {
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

############################
# 2) INITIATE SESSION
############################
def initiate_session(
    endpoint: str,
    tweaks: dict,
    stream: bool = False,
    api_key: Optional[str] = None,
):
    """
    POST to the HF Space run endpoint, requesting a stream if `stream=True`.
    Returns the JSON response with 'session_id' and possibly 'stream_url'.
    """
    url = f"{BASE_API_URL}/api/v1/run/{endpoint}?stream={str(stream).lower()}"

    payload = {
        "input_type": "chat",
        "output_type": "chat",
        "tweaks": tweaks,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    response = requests.post(url, json=payload, headers=headers, timeout=None)
    response.raise_for_status()
    return response.json()

############################
# 3) STREAMING CHUNKS
############################
def stream_chunks(stream_url: str, session_id: str, api_key: Optional[str] = None):
    full_url = f"{BASE_API_URL}{stream_url}"
    params = {"session_id": session_id}
    headers = {"x-api-key": api_key} if api_key else {}
    
    with requests.get(full_url, params=params, headers=headers, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                # Empty line (heartbeat). Skip it.
                continue

            decoded_line = line.decode("utf-8", errors="ignore").strip()
            # Only parse lines that start with "data:"
            if not decoded_line.startswith("data: "):
                # Could be "event:" or "retry:" lines
                continue

            # Remove 'data: ' prefix
            data_str = decoded_line[len("data: "):].strip()
            try:
                payload = json.loads(data_str)
            except json.JSONDecodeError:
                # Not valid JSON—possibly a partial line or SSE control line.
                continue

            chunk_text = payload.get("chunk")
            if chunk_text:
                yield chunk_text

############################
# 4) HIGH-LEVEL RUN (STREAM)
############################
def run_flow_stream_in_chunks(
    endpoint: str,
    tweaks: dict,
    api_key: Optional[str] = None,
):
    """
    Combines steps:
    1) initiate_session with stream=True
    2) if successful, parse the returned JSON for `stream_url`
    3) call `stream_chunks` to yield partial text
    """
    init_json = initiate_session(endpoint, tweaks, stream=True, api_key=api_key)

    # Typically we have: init_json["session_id"], init_json["outputs"][0]["outputs"][0]["artifacts"]["stream_url"]
    session_id = init_json.get("session_id", "")
    if not session_id:
        raise ValueError("No session_id returned in init response")

    # Dig out the first artifact with a stream_url
    stream_url = None
    outputs = init_json.get("outputs", [])
    if outputs and "outputs" in outputs[0]:
        first_output = outputs[0]["outputs"][0]
        artifacts = first_output.get("artifacts", {})
        stream_url = artifacts.get("stream_url")
    if not stream_url:
        # That means your flow might not have produced a stream_url
        # Possibly your LLM node isn't set to "stream=True"
        raise ValueError("No 'stream_url' found in the init JSON. Make sure your flow is streaming.")

    # Now stream partial text
    for chunk in stream_chunks(stream_url, session_id, api_key=api_key):
        yield chunk

############################
# 5) STREAMLIT UI
############################
st.set_page_config(page_icon="🌍", layout="wide")
st.title("Langflow + HF Space: Streaming Example with TWEAKS")

with st.sidebar:
    st.header("Tweak the input")
    with st.form("my_form"):
        company = st.text_input("Enter a company name:", "vonovia")
        submitted = st.form_submit_button("Run Flow")

if submitted:
    # 1) Update TWEAKS with the user's input
    TWEAKS["ChatInput-PKiJr"]["input_value"] = company
    # If your flow also uses "Prompt-WAIhH" -> "company", then set TWEAKS["Prompt-WAIhH"]["company"] = company

    # 2) Provide some streaming UI
    st.write("**Streaming partial response**:")
    chunk_container = st.empty()  # a placeholder to show partial text
    full_response = ""

    try:
        for chunk_text in run_flow_stream_in_chunks(
            endpoint=ENDPOINT or FLOW_ID,
            tweaks=TWEAKS,
            api_key=API_KEY,
        ):
            # Accumulate the chunk
            full_response += chunk_text
            # Show partial response in real-time
            chunk_container.markdown(f"```\n{full_response}\n```")
        st.success("Streaming complete!")
    except Exception as e:
        st.error(f"Error while streaming: {e}")

    # Optionally, parse JSON from final response if your flow returns a final JSON.
    # But if it's purely chunk-based, you already have everything in `full_response`.

