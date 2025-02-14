import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

# Custom CSS styling
st.markdown("""
<style>
    /* Global styles */
    .main {
        background-color: #f8f9ff;
        color: #4a5568;
        
    }

    /* Headers */
    h1, h2, h3 {
        color: #647acb !important;
        
        letter-spacing: -0.025em !important;
    }

    h1 {
        font-size: 2.25rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Text input styling */
    .stTextInput textarea {
        color: #4a5568 !important;
        background-color: #ffffff !important;
        border: 1px solid #e6eeff !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
        transition: all 0.15s ease-in-out !important;
    }

    .stTextInput textarea:focus {
        border-color: #a5b4fc !important;
        box-shadow: 0 0 0 3px rgba(165, 180, 252, 0.2) !important;
        outline: none !important;
    }

    /* Select box styling */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #ffffff !important;
        border: 1px solid #e6eeff !important;
        border-radius: 0.5rem !important;
        color: #4a5568 !important;
        font-size: 0.95rem !important;
    }

    .stSelectbox div[data-baseweb="select"]:hover {
        border-color: #a5b4fc !important;
    }

    /* Dropdown items */
    div[role="listbox"] div {
        background-color: #ffffff !important;
        color: #4a5568 !important;
        padding: 0.5rem 1rem !important;
    }

    div[role="listbox"] div:hover {
        background-color: #f0f4ff !important;
    }

    /* Chat message containers */
    .stChatMessage {
        background-color: #ffffff !important;
        border: 1px solid #e6eeff !important;
        border-radius: 1rem !important;
        padding: 1rem !important;
        margin: 0.75rem 0 !important;
        box-shadow: 0 2px 4px rgba(165, 180, 252, 0.1) !important;
    }

    /* User message specific styling */
    .stChatMessage[data-testid*="user"] {
        background-color: #f0f4ff !important;
    }

    /* Assistant message specific styling */
    .stChatMessage[data-testid*="assistant"] {
        background-color: #ffffff !important;
    }

    /* Code blocks */
    pre {
        background-color: #f8f9ff !important;
        border: 1px solid #e6eeff !important;
        border-radius: 0.5rem !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
    }

    code {
        color: #4a5568 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Buttons */
    .stButton button {
        background-color: #a5b4fc !important;
        color: #4a5568 !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.15s ease-in-out !important;
    }

    .stButton button:hover {
        background-color: #818cf8 !important;
        transform: translateY(-1px) !important;
    }

    /* Sidebar Container */
    .css-1d391kg {
        background-color: #f0f4ff !important;
        border-right: 1px solid #e6eeff !important;
        padding: 2rem 1rem !important;
    }

    /* Sidebar Title */
    .css-1d391kg h1 {
        color: #647acb !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin-bottom: 2rem !important;
    }

    /* Sidebar Headers */
    .css-1d391kg h2, .css-1d391kg h3 {
        color: #647acb !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }

    /* Sidebar Text */
    .css-1d391kg p {
        color: #4a5568 !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
    }

    /* Sidebar List Items */
    .css-1d391kg li {
        color: #4a5568 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
        list-style-type: none !important;
        padding-left: 0.5rem !important;
    }

    /* Links */
    a {
        color: #818cf8 !important;
        text-decoration: none !important;
        transition: color 0.15s ease-in-out !important;
    }

    a:hover {
        color: #6366f1 !important;
        text-decoration: underline !important;
    }

    /* Divider */
    hr {
        border: none !important;
        border-top: 1px solid #e6eeff !important;
        margin: 1.5rem 0 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f8f9ff;
    }

    ::-webkit-scrollbar-thumb {
        background: #c7d2fe;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #a5b4fc;
    }

    /* Status Messages */
    .stAlert {
        background-color: #fff1f2 !important;
        color: #be185d !important;
        border: 1px solid #fecdd3 !important;
        border-radius: 0.5rem !important;
    }

    .stInfo {
        background-color: #f0f9ff !important;
        color: #075985 !important;
        border: 1px solid #bae6fd !important;
        border-radius: 0.5rem !important;
    }

    .stSuccess {
        background-color: #f0fdf4 !important;
        color: #166534 !important;
        border: 1px solid #bbf7d0 !important;
        border-radius: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Page title
st.title("🐋 DeepSeek CodeAid")
st.caption("Your AI-Powered Debugging Wizard for Effortless Coding!")

# Sidebar configuration
with st.sidebar:
    st.markdown('<h2 class="sidebar-title">⚙️ Configuration</h2>', unsafe_allow_html=True)
    
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b"],
        index=0
    )
    
    st.markdown('<hr/>', unsafe_allow_html=True)
    
    st.markdown('<h3>💡 Quick Coding Tips</h3>', unsafe_allow_html=True)
    st.markdown("""
    <ul class="sidebar-list">
        <li>🛠️ Use meaningful variable names for better readability.</li>
        <li>🐞 Debug step by step instead of guessing the issue.</li>
        <li>⚡ Optimize loops to improve performance.</li>
        <li>📜 Write comments to explain complex logic.</li>
        <li>🔗 Keep functions small and modular for reusability.</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown('<hr/>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
    Built with 
    <a href="https://ollama.ai/" target="_blank">Ollama</a> | 
    <a href="https://python.langchain.com/" target="_blank">LangChain</a>
    </div>
    """, unsafe_allow_html=True)

# Enhanced system prompt
SYSTEM_TEMPLATE = SYSTEM_TEMPLATE = """You are an expert AI coding assistant specializing in Python programming. 
Your responses should:
1. Always include actual code implementations
2. Be clear and concise
3. Include helpful comments in the code
4. Add print statements for debugging when relevant
5. Explain the code briefly after showing it

When asked for code examples, always respond with actual working code first, then explain.
Do not include phrases like "I'm DeepSeek" or general statements about Python - focus on providing 
actual code solutions. Please hide the thinking part.

Example format:
```python
# Your code here
```

Here's how this code works:
[Brief explanation]


"""

# initialize LLM with error handling
try:
    llm_engine = ChatOllama(
        model=selected_model,
        base_url="http://localhost:11434",
        temperature=0.3
        #streaming=True
    )
except Exception as e:
    st.error(f"Error connecting to Ollama: {str(e)}")
    st.info("Please make sure Ollama is running and the model is installed.")
    st.stop()

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE)

# initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

# chat container
chat_container = st.container()

# display chat messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def generate_ai_response(prompt_chain):
    
    # for printing output line by line 
    #try:
        #processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
        
        # Create a placeholder for the streaming response
        #response_placeholder = st.empty()
        #full_response = ""
        
        # Process the stream
        #for chunk in processing_pipeline.stream({}):
            #full_response += chunk
            # Update the placeholder with the accumulated response
            #response_placeholder.markdown(full_response + "▌")
        
        # Return the complete response
        #return full_response
    try:
        processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
        return processing_pipeline.invoke({})
    except Exception as e:
        return f"Error generating response: {str(e)}\nPlease make sure Ollama is running and try again."

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.messages:
        content = msg["content"]
        # Escape curly braces in the content by doubling them
        escaped_content = content.replace("{", "{{").replace("}", "}}")
        
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(escaped_content))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(escaped_content))
    return ChatPromptTemplate.from_messages(prompt_sequence)

# chat input and processing
user_query = st.chat_input("Type your coding question here...")

if user_query:
    # add user message to log
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # generate AI response
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # add AI response to log
    st.session_state.messages.append({"role": "ai", "content": ai_response})
    
    # rerun to update chat display
    st.rerun()
