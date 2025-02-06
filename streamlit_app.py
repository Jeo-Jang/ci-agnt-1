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
ASTRA_KEY = st.secrets["ASTRA_DB_APP_TOKEN"]
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
TAVILY_KEY = st.secrets["TAVILY_API_KEY"]

# Define your API and flow settings
BASE_API_URL = "https://jeo-jang-langflownew.hf.space"
FLOW_ID = "9c281e13-bba9-49c3-90a3-83ccdacba39a" # Change new FLOW ID if the app is rebuilt
ENDPOINT = ""  # Set to empty string if not using a specific endpoint

# Define your tweaks dictionary. Notice that we set the user input to an empty string.
# Later, we update it with the value from the text input. Please see below with TWEAKS[blah blah][blah blah]
# TWEAKS = {
#     "ChatInput-8zFTw": {
#         "background_color": "",
#         "chat_icon": "",
#         "files": "",
#         "input_value": "",  #---!----This will be updated with the user's input------------!
#         "sender": "User",
#         "sender_name": "User",
#         "session_id": "",
#         "should_store_message": True,
#         "text_color": ""
#     },
#----------Change the TWEAK of the OpenAIModel too!----------------!
    # "OpenAIModel-W0BHu": {
    #     "api_key": {
    #         "load_from_db": False,
    #         "value": st.secrets["OPENAI_API_KEY"] #os.getenv("OPENAI_API_KEY")
    #     },


TWEAKS = {
  "ChatInput-PKiJr": {
    "files": "",
    "background_color": "",
    "chat_icon": "",
    "input_value": "", #---!----This will be updated with the user's input------------!
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
            "value": OPENAI_KEY #os.getenv("OPENAI_API_KEY")
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
    "stream": False,
    "system_message": "",
    "temperature": 0.1
  },
  "Prompt-coXDM": {
    "template": "\n\nRESEARCH PLAN: {previous_response}\n\nUse Tavily Search to investigate the queries and analyse the findings\nFocus on academic and reliable sources.\n\nSteps:\n1. Search using provided queries\n2. Analyse search results\n3. Verify source credibility\n4. Extract key findings\n\nFormat findings as:\nTitle: [Company's name] Circular End Target Category 1 Efficiency and Optimisation\n\nEvaluating the Each Targets into either:\n-Quantitative: \n-Qualitative: \n-No pledge: \nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\n\nProvide evidence (quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular format]:\n-Targets Names\n-Targets Descriptions\n-Target Values (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factors (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
    "tool_placeholder": "",
    "previous_response": ""
  },
  "Prompt-E4KTI": {
    "template": "You are an expert on company's sustainability report and it circular economy efforts and impact.\nYou will get to search for this information on {company}'s sustainability report. Please be neutral and scrutinise the information.\n\n2.Design for Circularity\nDefinition: Company starts using circular materials like recycled plastic, steel, aluminium; and begins to change the design of their products to create the foundation for a circular economy. \nExample Phrases:  \"modular parts and design\", \"easy to disassemble\", \"easy to recycle\", \"Our product is 100% recyclable\", \"we have used 50% of recycled aluminum\", \"Our product has now 10% longer life span, increased durability\", \"We replaced single-use parts with reusable alternatives\", \"Our packaging is entirely biodegradable\", \"We developed a cradle-to-cradle approach in product design\"\n\nIdentify Targets:\nUse the example phrases as  a guide to identify their targets related. DO NOT present this example these phrases as evidences.\n\nFor each Targets,\nEvaluating the Targets:\n-Quantitative: There is a clear metric or deadline.\n-Qualitative: A stated ambition without firm timelines or numbers.\n-No pledge: No mention of any circular end target.\nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\nProvide evidence (quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular data of targets]:\n-Targets Name\n-Targets Detailed Description\n-Target Value (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factor (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
    "tool_placeholder": "",
    "company": ""
  },
  "Agent-36xU0": {
    "add_current_date_tool": True,
    "agent_description": "A helpful assistant with access to the following tools:",
    "agent_llm": "OpenAI",
    "api_key": {
            "load_from_db": False,
            "value": OPENAI_KEY #os.getenv("OPENAI_API_KEY")
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
  "Prompt-8ONrC": {
    "template": "You are an expert research assistant.\n\nCreate a focused research plan that will guide our search.\n\nFormat your response exactly as:\n\nRESEARCH OBJECTIVE:\n[Clear statement of research goal]\n\nKEY SEARCH QUERIES:\n1. [Primary academic search query]\n2. [Secondary search query]\n3. [Alternative search approach]\n\nSEARCH PRIORITIES:\n- [What types of sources to focus on]\n- [Key aspects to investigate]\n- [Specific areas to explore]",
    "tool_placeholder": ""
  },
  "OpenAIModel-0LBLg": {
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
  "Prompt-tY383": {
    "template": "\n\nRESEARCH PLAN: {previous_response}\n\nUse Tavily Search to investigate the queries and analyse the findings\nFocus on academic and reliable sources.\n\nSteps:\n1. Search using provided queries\n2. Analyse search results\n3. Verify source credibility\n4. Extract key findings\n\nFormat findings as:\nTitle: [Company's name] Circular End Target Category 2 Design for Circularity\n\nEvaluating the Each Targets into either:\n-Quantitative: \n-Qualitative: \n-No pledge: \nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\nProvide evidence (quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular format]:\n-Targets Names\n-Targets Descriptions\n-Target Values (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factors (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
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
    "token": ASTRA_KEY
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
            "value": OPENAI_KEY #os.getenv("OPENAI_API_KEY")
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
  "ParseData-4nPAS": {
    "sep": "\n",
    "template": "{text}"
  },
  "Prompt-Vbv5R": {
    "template": "{context}\n\n---\n\nGiven the context above, answer the question as best as possible.\nHOWEVER, If there is no findings, please leave it empty and state there is no data on the database and additional web search data will complete this.\nName the Title as: [Title, vector database]\n\n\nQuestion: {question}\n\nAnswer: ",
    "tool_placeholder": "",
    "context": "",
    "question": ""
  },
  "AstraDB-oBJJI": {
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
    "token": ASTRA_KEY
  },
  "OpenAIEmbeddings-OQ2Pp": {
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
            "value": OPENAI_KEY #os.getenv("OPENAI_API_KEY")
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
  "Prompt-jdBR3": {
    "template": "You are an expert on company's sustainability report and it circular economy efforts and impact.\nYou will get to search for this information on {company}'s sustainability report. Please be neutral and scrutinise the information\n\n2.Design for Circularity\nDefinition: Company starts using circular materials like recycled plastic, steel, aluminium; and begins to change the design of their products to create the foundation for a circular economy. \nExample Phrases:  \"modular parts and design\", \"easy to disassemble\", \"easy to recycle\", \"Our product is 100% recyclable\", \"we have used 50% of recycled aluminum\", \"Our product has now 10% longer life span, increased durability\", \"We replaced single-use parts with reusable alternatives\", \"Our packaging is entirely biodegradable\", \"We developed a cradle-to-cradle approach in product design\"\n\nIdentify Targets:\nUse the example phrases as  a guide to identify their targets related. DO NOT present this example these phrases as evidences.\n\nFor each Targets,\nEvaluating the Targets:\n-Quantitative: There is a clear metric or deadline.\n-Qualitative: A stated ambition without firm timelines or numbers.\n-No pledge: No mention of any circular end target.\nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\nProvide evidence (quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular data of targets]:\n-Targets Name\n-Targets Detailed Description\n-Target Value (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factor (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
    "tool_placeholder": "",
    "company": ""
  },
  "Agent-4QsVh": {
    "add_current_date_tool": True,
    "agent_description": "A helpful assistant with access to the following tools:",
    "agent_llm": "OpenAI",
    "api_key": {
            "load_from_db": False,
            "value": OPENAI_KEY #os.getenv("OPENAI_API_KEY")
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
  "OpenAIModel-mCWYv": {
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
  "Prompt-L9FUb": {
    "template": "You are an expert on circular economy and sustainability reports.\nPlease combine these two information, do not simplify it.\n\nFormat must contain a table where you can see all the relevant information.\nPlease Make sure that the title is formatted in H1, named as Target 2: Design for Circularity, [company]",
    "tool_placeholder": ""
  },
  "CombineText-Ealnb": {
    "delimiter": " _____",
    "text1": "",
    "text2": ""
  },
  "Prompt-6Lf7y": {
    "template": "You are an expert on company's sustainability report and it circular economy efforts and impact.\nYou will get to search for this information on {company}'s sustainability report. Please be neutral and scrutinise the information.\n\n3.Establish Processes to Take Back Products or Parts\nDefinition: Company creates processes to get hands on their products again during or after use phase. R-Strategies apply: Take Back, Recycling, Remanufacturing, Repair. \nExample Phrases: \"We collect or takeback XYZ\", \"We take the responsibility for our products/service\", \"Chemical recycling of our plastics\", \"owns or support the material recovery facilities...\" \"We provide incentives for customers to return used products\", \"We established a repair program for end-of-life devices\", \"We partner with third-party recyclers to ensure closed-loop material flows\", \"We maintain a buy-back system for outdated models\"\n\nIdentify Targets:\nUse the example phrases as  a guide to identify their targets related. DO NOT present this example these phrases as evidences.\n\nFor each Targets,\nEvaluating the Targets:\n-Quantitative: There is a clear metric or deadline.\n-Qualitative: A stated ambition without firm timelines or numbers.\n-No pledge: No mention of any circular end target.\nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\nProvide evidence (quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular data of targets]:\n-Targets Name\n-Targets Detailed Description\n-Target Value (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factor (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
    "tool_placeholder": "",
    "company": ""
  },
  "Agent-mlx5x": {
    "add_current_date_tool": True,
    "agent_description": "A helpful assistant with access to the following tools:",
    "agent_llm": "OpenAI",
    "api_key": {
            "load_from_db": False,
            "value": OPENAI_KEY #os.getenv("OPENAI_API_KEY")
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
  "Prompt-aF8Sb": {
    "template": "You are an expert research assistant.\n\nCreate a focused research plan that will guide our search.\n\nFormat your response exactly as:\n\nRESEARCH OBJECTIVE:\n[Clear statement of research goal]\n\nKEY SEARCH QUERIES:\n1. [Primary academic search query]\n2. [Secondary search query]\n3. [Alternative search approach]\n\nSEARCH PRIORITIES:\n- [What types of sources to focus on]\n- [Key aspects to investigate]\n- [Specific areas to explore]",
    "tool_placeholder": ""
  },
  "OpenAIModel-ylJtq": {
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
  "Prompt-mlqkj": {
    "template": "\n\nRESEARCH PLAN: {previous_response}\n\nUse Tavily Search to investigate the queries and analyse the findings\nFocus on academic and reliable sources.\n\nSteps:\n1. Search using provided queries\n2. Analyse search results\n3. Verify source credibility\n4. Extract key findings\n\nFormat findings as:\nTitle: [Company's name] Circular End Target Category 3.Establish Processes to Take Back Products or Parts\n\nEvaluating the Each Targets into either:\n-Quantitative: \n-Qualitative: \n-No pledge: \nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\nProvide evidence (quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular format]:\n-Targets Names\n-Targets Descriptions\n-Target Values (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factors (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
    "tool_placeholder": "",
    "previous_response": ""
  },
  "ParseData-UW9z0": {
    "sep": "\n",
    "template": "{text}"
  },
  "Prompt-kBDrX": {
    "template": "{context}\n\n---\n\nGiven the context above, answer the question as best as possible.\nHOWEVER, If there is no findings, please leave it empty and state there is no data on the database and additional web search data will complete this.\nName the Title as: [Title, vector database]\n\n\nQuestion: {question}\n\nAnswer: ",
    "tool_placeholder": "",
    "context": "",
    "question": ""
  },
  "AstraDB-YYFVJ": {
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
    "token": ASTRA_KEY
  },
  "OpenAIEmbeddings-vkK9L": {
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
            "value": OPENAI_KEY #os.getenv("OPENAI_API_KEY")
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
  "Prompt-Pp5Yc": {
    "template": "You are an expert on company's sustainability report and it circular economy efforts and impact.\nYou will get to search for this information on {company}'s sustainability report. Please be neutral and scrutinise the information\n\n3.Establish Processes to Take Back Products or Parts\nDefinition: Company creates processes to get hands on their products again during or after use phase. R-Strategies apply: Take Back, Recycling, Remanufacturing, Repair. \nExample Phrases: \"We collect or takeback XYZ\", \"We take the responsibility for our products/service\", \"Chemical recycling of our plastics\", \"owns or support the material recovery facilities...\" \"We provide incentives for customers to return used products\", \"We established a repair program for end-of-life devices\", \"We partner with third-party recyclers to ensure closed-loop material flows\", \"We maintain a buy-back system for outdated models\"\n\nIdentify Targets:\nUse the example phrases as  a guide to identify their targets related. DO NOT present this example these phrases as evidences.\n\nFor each Targets,\nEvaluating the Targets:\n-Quantitative: There is a clear metric or deadline.\n-Qualitative: A stated ambition without firm timelines or numbers.\n-No pledge: No mention of any circular end target.\nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\nProvide evidence (quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular data of targets]:\n-Targets Name\n-Targets Detailed Description\n-Target Value (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factor (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
    "tool_placeholder": "",
    "company": ""
  },
  "Agent-rU34U": {
    "add_current_date_tool": True,
    "agent_description": "A helpful assistant with access to the following tools:",
    "agent_llm": "OpenAI",
    "api_key": {
            "load_from_db": False,
            "value": OPENAI_KEY #os.getenv("OPENAI_API_KEY")
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
  "OpenAIModel-49lhr": {
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
  "Prompt-A5ePq": {
    "template": "You are an expert on circular economy and sustainability reports.\nPlease combine these two information, do not simplify it.\n\nFormat must contain a table where you can see all the relevant information.\nPlease Make sure that the title is formatted in H1, named as Target 3: Establish Processes to Take Back Products or Parts, [company]",
    "tool_placeholder": ""
  },
  "CombineText-lPsUz": {
    "delimiter": " _____",
    "text1": "",
    "text2": ""
  },
  "Prompt-dCnKX": {
    "template": "You are an expert on company's sustainability report and it circular economy efforts and impact.\nYou will get to search for this information on {company}'s sustainability report. Please be neutral and scrutinise the information.\n\n4.Earn Money with Circular Strategies\nDefinition: Company uses R-Strategies as a fundamental part of their business to strengthen circularity like Remanufacturing, Product as a Service, Upgrades, with the goal to close, narrow or slow down loops. R-Strategies can be combined: Refurbish, Remanufacture, Upgrade, Repair, Reuse, collaborate. High commitment to show responsibility for own products and materials with the goal to retain value as long as possible.\nExample Phrases: \"We establish a second market for used goods\", \"We reuse our products as long es possible\", \"We change our business model to offer services instead of products\", \"We implement remanufacturing lines to refurbish older units\", \"We generate revenue by leasing equipment rather than selling it\", \"We monetize upgraded components to extend product life cycles\", \"We invest in product-as-a-service platforms to retain ownership of materials\"\n\nIdentify Targets:\nUse the example phrases as  a guide to identify their targets related. DO NOT present this example these phrases as evidences.\n\nFor each Targets,\nEvaluating the Targets:\n-Quantitative: There is a clear metric or deadline.\n-Qualitative: A stated ambition without firm timelines or numbers.\n-No pledge: No mention of any circular end target.\nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\nProvide evidence (quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular data of targets]:\n-Targets Name\n-Targets Detailed Description\n-Target Value (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factor (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
    "tool_placeholder": "",
    "company": ""
  },
  "Agent-4suN6": {
    "add_current_date_tool": True,
    "agent_description": "A helpful assistant with access to the following tools:",
    "agent_llm": "OpenAI",
    "api_key": {
            "load_from_db": False,
            "value": OPENAI_KEY #os.getenv("OPENAI_API_KEY")
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
  "Prompt-LQnzM": {
    "template": "You are an expert research assistant.\n\nCreate a focused research plan that will guide our search.\n\nFormat your response exactly as:\n\nRESEARCH OBJECTIVE:\n[Clear statement of research goal]\n\nKEY SEARCH QUERIES:\n1. [Primary academic search query]\n2. [Secondary search query]\n3. [Alternative search approach]\n\nSEARCH PRIORITIES:\n- [What types of sources to focus on]\n- [Key aspects to investigate]\n- [Specific areas to explore]",
    "tool_placeholder": ""
  },
  "OpenAIModel-u1Lal": {
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
  "Prompt-WtNh3": {
    "template": "\n\nRESEARCH PLAN: {previous_response}\n\nUse Tavily Search to investigate the queries and analyse the findings\nFocus on academic and reliable sources.\n\nSteps:\n1. Search using provided queries\n2. Analyse search results\n3. Verify source credibility\n4. Extract key findings\n\nFormat findings as:\nTitle: [Company's name] Circular End Target Category 4.Earn Money with Circular Strategies\n\nEvaluating the Each Targets into either:\n-Quantitative: \n-Qualitative: \n-No pledge: \nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\nProvide evidence (quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular format]:\n-Targets Names\n-Targets Descriptions\n-Target Values (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factors (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
    "tool_placeholder": "",
    "previous_response": ""
  },
  "ParseData-JVIbZ": {
    "sep": "\n",
    "template": "{text}"
  },
  "Prompt-dMQVo": {
    "template": "{context}\n\n---\n\nGiven the context above, answer the question as best as possible.\nHOWEVER, If there is no findings, please leave it empty and state there is no data on the database and additional web search data will complete this.\nName the Title as: [Title, vector database]\n\n\nQuestion: {question}\n\nAnswer: ",
    "tool_placeholder": "",
    "context": "",
    "question": ""
  },
  "AstraDB-Fcst3": {
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
    "token": ASTRA_KEY
  },
  "OpenAIEmbeddings-NIj9H": {
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
            "value": OPENAI_KEY #os.getenv("OPENAI_API_KEY")
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
  "Prompt-87hSl": {
    "template": "You are an expert on company's sustainability report and it circular economy efforts and impact.\nYou will get to search for this information on {company}'s sustainability report. Please be neutral and scrutinise the information\n\n4.Earn Money with Circular Strategies\nDefinition: Company uses R-Strategies as a fundamental part of their business to strengthen circularity like Remanufacturing, Product as a Service, Upgrades, with the goal to close, narrow or slow down loops. R-Strategies can be combined: Refurbish, Remanufacture, Upgrade, Repair, Reuse, collaborate. High commitment to show responsibility for own products and materials with the goal to retain value as long as possible.\nExample Phrases: \"We establish a second market for used goods\", \"We reuse our products as long es possible\", \"We change our business model to offer services instead of products\", \"We implement remanufacturing lines to refurbish older units\", \"We generate revenue by leasing equipment rather than selling it\", \"We monetize upgraded components to extend product life cycles\", \"We invest in product-as-a-service platforms to retain ownership of materials\"\n\nIdentify Targets:\nUse the example phrases as  a guide to identify their targets related. DO NOT present this example these phrases as evidences.\n\nFor each Targets,\nEvaluating the Targets:\n-Quantitative: There is a clear metric or deadline.\n-Qualitative: A stated ambition without firm timelines or numbers.\n-No pledge: No mention of any circular end target.\nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\nProvide evidence (quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular data of targets]:\n-Targets Name\n-Targets Detailed Description\n-Target Value (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factor (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
    "tool_placeholder": "",
    "company": ""
  },
  "Agent-3JXRU": {
    "add_current_date_tool": True,
    "agent_description": "A helpful assistant with access to the following tools:",
    "agent_llm": "OpenAI",
    "api_key": {
            "load_from_db": False,
            "value": OPENAI_KEY #os.getenv("OPENAI_API_KEY")
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
  "OpenAIModel-E4bwI": {
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
  "Prompt-u0JRf": {
    "template": "You are an expert on circular economy and sustainability reports.\nPlease combine these two information, do not simplify it.\n\nFormat must contain a table where you can see all the relevant information.\nPlease Make sure that the title is formatted in H1, named as Target 4: Earn Money with Circular Strategies, [company]",
    "tool_placeholder": ""
  },
  "CombineText-SCusB": {
    "delimiter": " _____",
    "text1": "",
    "text2": ""
  },
  "Prompt-cXev8": {
    "template": "You are an expert on company's sustainability report and it circular economy efforts and impact.\nYou will get to search for this information on {company}'s sustainability report. Please be neutral and scrutinise the information.\n\n5.Change Industry Towards a Circular Ecosystem.\nDefinition: Establish shared networks of resources and energy to close loops and circulate materials/energy/parts; share digital infrastructure like platforms, ai-systems, robotics. This can happen in the own industry or cross-industry \n\nExample Phrases: \"We work with suppliers  and clients to establish a circular system\", \"We collaborate with competitors to establish standards\", \"We co-develop recycling infrastructure with partners across industries\", \"We share data and technology to accelerate the circular transition\", \"We collaborate with local governments to create closed-loop solutions\", \"We standardize material specifications to enable cross-company reuse\"\n\nIdentify Targets:\nUse the example phrases as  a guide to identify their targets related. DO NOT present this example these phrases as evidences.\n\nFor each Targets,\nEvaluating the Targets:\n-Quantitative: There is a clear metric or deadline.\n-Qualitative: A stated ambition without firm timelines or numbers.\n-No pledge: No mention of any circular end target.\nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\nProvide evidence (quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular data of targets]:\n-Targets Name\n-Targets Detailed Description\n-Target Value (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factor (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
    "tool_placeholder": "",
    "company": ""
  },
  "Agent-UA26M": {
    "add_current_date_tool": True,
    "agent_description": "A helpful assistant with access to the following tools:",
    "agent_llm": "OpenAI",
    "api_key": {
            "load_from_db": False,
            "value": OPENAI_KEY #os.getenv("OPENAI_API_KEY")
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
  "Prompt-hzbft": {
    "template": "You are an expert research assistant.\n\nCreate a focused research plan that will guide our search.\n\nFormat your response exactly as:\n\nRESEARCH OBJECTIVE:\n[Clear statement of research goal]\n\nKEY SEARCH QUERIES:\n1. [Primary academic search query]\n2. [Secondary search query]\n3. [Alternative search approach]\n\nSEARCH PRIORITIES:\n- [What types of sources to focus on]\n- [Key aspects to investigate]\n- [Specific areas to explore]",
    "tool_placeholder": ""
  },
  "OpenAIModel-59onK": {
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
  "Prompt-AYR5y": {
    "template": "\n\nRESEARCH PLAN: {previous_response}\n\nUse Tavily Search to investigate the queries and analyse the findings\nFocus on academic and reliable sources.\n\nSteps:\n1. Search using provided queries\n2. Analyse search results\n3. Verify source credibility\n4. Extract key findings\n\nFormat findings as:\nTitle: [Company's name] Circular End Target Category 5.Change Industry Towards Circular Ecosystem\n\nEvaluating the Each Targets into either:\n-Quantitative: \n-Qualitative: \n-No pledge: \nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\nProvide evidence (quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular format]:\n-Targets Names\n-Targets Descriptions\n-Target Values (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factors (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
    "tool_placeholder": "",
    "previous_response": ""
  },
  "ParseData-7ceTq": {
    "sep": "\n",
    "template": "{text}"
  },
  "Prompt-Ze6JI": {
    "template": "{context}\n\n---\n\nGiven the context above, answer the question as best as possible.\nHOWEVER, If there is no findings, please leave it empty and state there is no data on the database and additional web search data will complete this.\nName the Title as: Company name, vector databse\n\n\nQuestion: {question}\n\nAnswer: ",
    "tool_placeholder": "",
    "context": "",
    "question": ""
  },
  "AstraDB-qEfVk": {
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
    "token": ASTRA_KEY
  },
  "OpenAIEmbeddings-Xs2c1": {
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
            "value": OPENAI_KEY #os.getenv("OPENAI_API_KEY")
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
  "Prompt-nKGjw": {
    "template": "You are an expert on company's sustainability report and it circular economy efforts and impact.\nYou will get to search for this information on {company}'s sustainability report. Please be neutral and scrutinise the information\n\n5.Change Industry Towards a Circular Ecosystem.\nDefinition: Establish shared networks of resources and energy to close loops and circulate materials/energy/parts; share digital infrastructure like platforms, ai-systems, robotics. This can happen in the own industry or cross-industry \n\nExample Phrases: \"We work with suppliers  and clients to establish a circular system\", \"We collaborate with competitors to establish standards\", \"We co-develop recycling infrastructure with partners across industries\", \"We share data and technology to accelerate the circular transition\", \"We collaborate with local governments to create closed-loop solutions\", \"We standardize material specifications to enable cross-company reuse\"\n\nIdentify Targets:\nUse the example phrases as  a guide to identify their targets related. DO NOT present this example these phrases as evidences.\n\nFor each Targets,\nEvaluating the Targets:\n-Quantitative: There is a clear metric or deadline.\n-Qualitative: A stated ambition without firm timelines or numbers.\n-No pledge: No mention of any circular end target.\nAssign a rating (5, 2, or 1) based on:\n-5: Clear, quantitative pledge for full circularity by a specific year.\n-2: Qualitative pledge without specific metrics or deadlines.\n-1: No mention of a circular end target at all.\nAssign the impact factor(IF) based on:\n-3: High, Global roll out of activities\n-2: Medium, limited to one product category or business unit or country\n-1: Low, Just on a pilot scale with no real impact compared to the companies revenue, co2 emission, etc.\n\n\nProvide evidence (quotes or references to the report) justifying your rating.\n\nCompile your findings with following format:\n[Company Name]\n[Tabular data of targets]:\n-Targets Name\n-Targets Detailed Description\n-Target Value (time and/or %)\n-Evidence Activities\n-Achieved Successes\n-The assigned Impact Factor (IF = 1, 2, or 3)\n[Explanation and Sources for IF ratings]",
    "tool_placeholder": "",
    "company": ""
  },
  "Agent-hM2MU": {
    "add_current_date_tool": True,
    "agent_description": "A helpful assistant with access to the following tools:",
    "agent_llm": "OpenAI",
    "api_key": {
            "load_from_db": False,
            "value": OPENAI_KEY #os.getenv("OPENAI_API_KEY")
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
  "OpenAIModel-dWbCV": {
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
  "Prompt-dzmsC": {
    "template": "You are an expert on circular economy and sustainability reports.\nPlease combine these two information, do not simplify it.\n\nFormat must contain a table where you can see all the relevant information.\nPlease Make sure that the title is formatted in H1, named as Target 5: Change Industry Towards a Circular Ecosystem, [company]",
    "tool_placeholder": ""
  },
  "CombineText-m818S": {
    "delimiter": " _____",
    "text1": "",
    "text2": ""
  },
  "CombineText-aiR0t": {
    "delimiter": " ______________________________________________________",
    "text1": "",
    "text2": ""
  },
  "CombineText-KxP43": {
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
    "stream": False,
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
    "max_results": 2, #default 5
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
  "TavilySearchComponent-Lt34r": {
    "api_key": {
        "load_from_db": False,
        "value": TAVILY_KEY
    },
    "include_answer": True,
    "include_images": True,
    "max_results": 2, #default 5
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
  "TavilySearchComponent-DLZrC": {
    "api_key": {
        "load_from_db": False,
        "value": TAVILY_KEY
    },
    "include_answer": True,
    "include_images": True,
    "max_results": 2, #default 5
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
  "TavilySearchComponent-cJDx1": {
    "api_key": {
        "load_from_db": False,
        "value": TAVILY_KEY
    },
    "include_answer": True,
    "include_images": True,
    "max_results": 2, #default 5
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
  "TavilySearchComponent-xSWhx": {
    "api_key": {
        "load_from_db": False,
        "value": TAVILY_KEY
    },
    "include_answer": True,
    "include_images": True,
    "max_results": 2, #default 5
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
            "value": OPENAI_KEY #os.getenv("OPENAI_API_KEY")
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


#----------------------Delete Below After Succssful Transition----------!!


# TWEAKS = {
#     "ChatInput-8zFTw": {
#         "background_color": "",
#         "chat_icon": "",
#         "files": "",
#         "input_value": "",  # This will be updated with the user's input
#         "sender": "User",
#         "sender_name": "User",
#         "session_id": "",
#         "should_store_message": True,
#         "text_color": ""
#     },
#     "TextInput-Q7ye5": {
#         "input_value": "- Thread must be 5-7 tweets long - Each tweet should be self-contained but flow naturally to the next - Include relevant technical details while keeping language accessible - Use emojis sparingly but effectively - Include a clear call-to-action in the final tweet - Highlight key benefits and innovative aspects - Maintain professional but engaging tone"
#     },
#     "ChatOutput-wTQTP": {
#         "background_color": "",
#         "chat_icon": "",
#         "data_template": "{text}",
#         "input_value": "",
#         "sender": "Machine",
#         "sender_name": "AI",
#         "session_id": "",
#         "should_store_message": True,
#         "text_color": ""
#     },
#     "TextInput-BQHnI": {
#         "input_value": "thread"
#     },
#     "TextInput-X18Sl": {
#         "input_value": "English"
#     },
#     "TextInput-VJ8Zw": {
#         "input_value": "- Tech startup focused on Vegan food"
#     },
#     "TextInput-DnWUD": {
#         "input_value": "- Professional yet approachable - annoyed and negative"
#     },
#     "TextInput-8FJGR": {
#         "input_value": "Vegan product Company"
#     },
#     "Prompt-rZryL": {
#         "CONTENT_GUIDELINES": "",
#         "OUTPUT_FORMAT": "",
#         "OUTPUT_LANGUAGE": "",
#         "PROFILE_DETAILS": "",
#         "PROFILE_TYPE": "",
#         "TONE_AND_STYLE": "",
#         "template": (
#             "<Instructions Structure>\nIntroduce the task of generating tweets or tweet threads based on the provided inputs\n\n"
#             "Explain each input variable:\n\n{{PROFILE_TYPE}}\n\n{{PROFILE_DETAILS}}\n\n{{CONTENT_GUIDELINES}}\n\n{{TONE_AND_STYLE}}\n\n"
#             "{{CONTEXT}}\n\n{{OUTPUT_FORMAT}}\n\n{{OUTPUT_LANGUAGE}}\n\n"
#             "Provide step-by-step instructions on how to analyze the inputs to determine if a single tweet or thread is appropriate\n\n"
#             "Give guidance on generating tweet content that aligns with the profile, guidelines, tone, style, and context\n\n"
#             "Explain how to format the output based on the {{OUTPUT_FORMAT}} value\n\n"
#             "Provide tips for creating engaging, coherent tweet content\n\n</Instructions Structure>\n\n"
#             "<Instructions>\nYou are an AI tweet generator that can create standalone tweets or multi-tweet threads based on a variety of inputs about the desired content. Here are the key inputs you will use to generate the tweet(s):\n\n"
#             "<profile_type>\n\n{PROFILE_TYPE}\n\n</profile_type>\n\n"
#             "<profile_details>\n\n{PROFILE_DETAILS}\n\n</profile_details>\n\n"
#             "<content_guidelines>\n\n{CONTENT_GUIDELINES}\n\n</content_guidelines>\n\n"
#             "<tone_and_style>\n\n{TONE_AND_STYLE}\n\n</tone_and_style>\n\n"
#             "<output_format>\n\n{OUTPUT_FORMAT}\n\n</output_format>\n\n"
#             "<output_language>\n\n{OUTPUT_LANGUAGE}\n\n</output_language>\n\n"
#             "To generate the appropriate tweet(s), follow these steps:\n\n"
#             "<output_determination>\n\nCarefully analyze the {{PROFILE_TYPE}}, {{PROFILE_DETAILS}}, {{CONTENT_GUIDELINES}}, {{TONE_AND_STYLE}}, and {{CONTEXT}} to determine the depth and breadth of content needed.\n\n"
#             "If the {{OUTPUT_FORMAT}} is \"single_tweet\", plan to convey the key information in a concise, standalone tweet.\n\n"
#             "If the {{OUTPUT_FORMAT}} is \"thread\" or if the content seems too complex for a single tweet, outline a series of connected tweets that flow together to cover the topic.\n\n"
#             "</output_determination>\n\n"
#             "<content_generation>\n\nBrainstorm tweet content that aligns with the {{PROFILE_TYPE}} and {{PROFILE_DETAILS}}, adheres to the {{CONTENT_GUIDELINES}}, matches the {{TONE_AND_STYLE}}, and incorporates the {{CONTEXT}}.\n\n"
#             "For a single tweet, craft the most engaging, informative message possible within the 280 character limit.\n\n"
#             "For a thread, break down the content into distinct yet connected tweet-sized chunks. Ensure each tweet flows logically into the next to maintain reader engagement. Use transitional phrases as needed to link tweets.\n\n"
#             "</content_generation>\n\n"
#             "<formatting>\nFormat the output based on the {{OUTPUT_FORMAT}}:\n\n"
#             "For a single tweet, provide the content.\n\n"
#             "For a thread, include each tweet inside numbered markdown list.\n\n"
#             "</formatting>\n\n"
#             "<tips>\nFocus on creating original, engaging content that provides value to the intended audience.\n\n"
#             "Optimize the tweet(s) for the 280 character limit. Be concise yet impactful.\n\n"
#             "Maintain a consistent voice that matches the {{TONE_AND_STYLE}} throughout the tweet(s).\n\n"
#             "Include calls-to-action or questions to drive engagement when appropriate.\n\n"
#             "Double check that the final output aligns with the {{PROFILE_DETAILS}} and {{CONTENT_GUIDELINES}}.\n\n"
#             "</tips>\n\nNow create a Tweet or Twitter Thread for this context:\n\n"
#         ),
#         "tool_placeholder": ""
#     },
#     "OpenAIModel-W0BHu": {
#         "api_key": {
#             "load_from_db": False,
#             "value": st.secrets["OPENAI_API_KEY"] #os.getenv("OPENAI_API_KEY")
#         },
#         "input_value": "",
#         "json_mode": False,
#         "max_tokens": None,
#         "model_kwargs": {},
#         "model_name": "gpt-4o-mini",
#         "openai_api_base": "",
#         "seed": 1,
#         "stream": False,
#         "system_message": "",
#         "temperature": 0.1
#     }
# }

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
user_message = st.text_input("Enter company name:")


# If you want to allow file upload as well, you can use st.file_uploader (optional)
# uploaded_file = st.file_uploader("Upload a file", type=["txt", "csv", "json"])

if st.button("Submit"):
    if user_message:
        # Update the tweak for the chat input with the user's message
        TWEAKS["ChatInput-PKiJr"]["input_value"] = user_message


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
