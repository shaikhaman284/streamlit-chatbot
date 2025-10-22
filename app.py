import streamlit as st
import tempfile
import os
from datetime import datetime
import math
import re
import json
import time
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

# Import fixes
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError as e:
    st.error(f"❌ Missing packages: {e}")
    st.stop()

st.set_page_config(page_title="AI Agent with PDF Q&A", layout="wide")
st.title("🤖 AI Agent with PDF Q&A & Calculator (Day 7)")

# Create data directory for persistent storage
DATA_DIR = "data"
VECTORSTORE_DIR = os.path.join(DATA_DIR, "vectorstores")
CHAT_EXPORTS_DIR = os.path.join(DATA_DIR, "chat_exports")

for dir_path in [DATA_DIR, VECTORSTORE_DIR, CHAT_EXPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ---------- SESSION ----------
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Persistence Functions ----------
def get_pdf_hash(uploaded_file):
    """Generate a hash for the PDF file content"""
    content = uploaded_file.getbuffer()
    return hashlib.md5(content).hexdigest()

def get_vectorstore_path(pdf_hash, chunk_size, chunk_overlap):
    """Get the path for saving/loading vectorstore"""
    return os.path.join(VECTORSTORE_DIR, f"vs_{pdf_hash}_{chunk_size}_{chunk_overlap}")

def save_vectorstore(vectorstore, path):
    """Save vectorstore to disk"""
    try:
        vectorstore.save_local(path)
        return True
    except Exception as e:
        st.error(f"Failed to save vectorstore: {e}")
        return False

def load_vectorstore(path):
    """Load vectorstore from disk"""
    try:
        embeddings = load_embedding_model()
        vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    except Exception as e:
        return None

# ---------- Streaming Functions ----------
def stream_text(text, container, delay=0.03):
    """Display text with typing animation"""
    displayed_text = ""
    for char in text:
        displayed_text += char
        container.markdown(displayed_text)
        time.sleep(delay)

def simulate_streaming_response(text, container):
    """Simulate streaming response with typing effect"""
    words = text.split()
    displayed_text = ""
    
    for i, word in enumerate(words):
        displayed_text += word + " "
        container.markdown(displayed_text)
        time.sleep(0.05)  # Adjust speed as needed

# ---------- Export Functions ----------
def export_chat_as_markdown():
    """Export chat history as markdown"""
    if not st.session_state.messages:
        return "No chat history to export."
    
    markdown_content = f"# Chat Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    if st.session_state.user_name:
        markdown_content += f"**User:** {st.session_state.user_name}\n\n"
    
    for msg in st.session_state.messages:
        role = "**User**" if msg["role"] == "user" else "**AI Assistant**"
        markdown_content += f"{role}: {msg['content']}\n\n---\n\n"
    
    return markdown_content

def export_chat_as_json():
    """Export chat history as JSON"""
    chat_data = {
        "export_timestamp": datetime.now().isoformat(),
        "user_name": st.session_state.user_name,
        "pdf_name": st.session_state.get("pdf_name"),
        "messages": st.session_state.messages
    }
    return json.dumps(chat_data, indent=2)

# ---------- Chat Context Memory Helper ----------
def get_conversation_context(n=3):
    """Collect last N Q&A turns for prompt context."""
    history = st.session_state.get("messages", [])
    context = ""
    for msg in history[-n*2:]:
        speaker = "User" if msg["role"] == "user" else "AI"
        context += f"{speaker}: {msg['content']}\n"
    return context.strip()

def extract_name_from_message(message):
    patterns = [
        r"my name is (\w+)",
        r"i am (\w+)",
        r"i'm (\w+)",
        r"call me (\w+)",
        r"name's (\w+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, message.lower())
        if match:
            return match.group(1).capitalize()
    return None

def is_simple_greeting(message):
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'howdy']
    return message.lower().strip() in greetings

def handle_simple_query_directly(user_input):
    user_input_lower = user_input.lower().strip()
    if is_simple_greeting(user_input):
        if st.session_state.user_name:
            return f"Hello {st.session_state.user_name}! 👋 How can I help you today?"
        else:
            return "Hello! 👋 How can I help you today? You can ask me questions about PDFs, math problems, or general topics."
    if any(phrase in user_input_lower for phrase in ['what can you do', 'help me', 'what are your capabilities']):
        return """I'm an AI agent that can help you with:

📄 **PDF Analysis** - Upload a PDF and ask questions about its content
🧮 **Calculations** - Solve math problems like "What is 15 * 24?"
💬 **General Questions** - Answer questions on various topics

Try asking me something like:
- "What is 25 + 17?"  
- "Summarize my PDF document"
- "What is artificial intelligence?"

How can I assist you today?"""
    return None

# ---------- Enhanced Tool Functions ----------
def calculator_tool_func(expression: str) -> str:
    try:
        expression = expression.replace("×", "*").replace("÷", "/").replace("x", "*").strip()
        allowed_chars = set("0123456789+-*/().= ")
        if not all(c in allowed_chars for c in expression):
            return "ERROR: Only basic math operations allowed."
        if '=' in expression:
            expression = expression.split('=')[0].strip()
        result = eval(expression)
        return f"The calculation result is: {result}"
    except Exception as e:
        return f"ERROR: Could not calculate '{expression}' - {str(e)}"

def rag_tool_func(query: str) -> str:
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        return "No PDF document has been uploaded yet. Please upload a PDF file first to ask questions about its content."
    try:
        vectorstore = st.session_state.vectorstore
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "I couldn't find relevant information in the uploaded PDF."
        context_parts = []
        for i, doc in enumerate(docs[:4]):
            metadata = getattr(doc, 'metadata', {})
            page = metadata.get('page', i+1)
            content = doc.page_content.strip()
            context_parts.append(f"[Page {page}]: {content}")
        pdf_context = "\n\n".join(context_parts)
        llm = st.session_state.get("llm")
        if llm:
            name_context = f"The user's name is {st.session_state.user_name}. " if st.session_state.user_name else ""
            convo_context = get_conversation_context(n=3)
            prompt = f"""{name_context}
Past conversation:
{convo_context}

You are an AI assistant helping with PDF document analysis. Based on the following excerpts from the PDF, answer the user's new question. Be specific and helpful.

PDF Content:
{pdf_context[:2500]}

User: {query}

Answer:"""
            response = llm.invoke(prompt)
            return response.content
        else:
            return f"Based on the PDF document:\n\n{pdf_context[:1000]}..."
    except Exception as e:
        return f"I encountered an error while searching the PDF document: {str(e)}. Please try rephrasing your question."

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def process_uploaded_pdf(uploaded_file, chunk_size, chunk_overlap):
    """Process PDF with disk persistence"""
    filename = uploaded_file.name
    pdf_hash = get_pdf_hash(uploaded_file)
    vectorstore_path = get_vectorstore_path(pdf_hash, chunk_size, chunk_overlap)
    cache_key = f"{filename}_{pdf_hash}_{chunk_size}_{chunk_overlap}"
    
    # Try to load from disk first
    if os.path.exists(vectorstore_path):
        with st.spinner("📂 Loading cached PDF index from disk..."):
            vectorstore = load_vectorstore(vectorstore_path)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.cache_key = cache_key
                st.session_state.pdf_name = filename
                st.session_state.pdf_hash = pdf_hash
                st.success("✅ Loaded cached PDF index from disk!")
                return True
    
    # Process PDF if not cached or loading failed
    if st.session_state.get("cache_key") != cache_key:
        with st.spinner("📚 Processing PDF and saving to disk..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                if not pages:
                    st.error("❌ Could not extract text from PDF")
                    try: os.remove(tmp_path)
                    except: pass
                    return False
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                docs = text_splitter.split_documents(pages)
                embeddings = load_embedding_model()
                vectorstore = FAISS.from_documents(docs, embeddings)
                
                # Save to session state
                st.session_state.vectorstore = vectorstore
                st.session_state.cache_key = cache_key
                st.session_state.pdf_name = filename
                st.session_state.pdf_hash = pdf_hash
                
                # Save to disk
                if save_vectorstore(vectorstore, vectorstore_path):
                    st.success(f"✅ PDF processed and saved to disk! {len(pages)} pages, {len(docs)} chunks")
                else:
                    st.warning("⚠️ PDF processed but couldn't save to disk")
                
                try: os.remove(tmp_path)
                except: pass
                return True
                
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                try:
                    if 'tmp_path' in locals():
                        os.remove(tmp_path)
                except: pass
                return False
    else:
        if "vectorstore" in st.session_state and st.session_state.vectorstore is not None:
            st.info("ℹ️ Using cached PDF data - ready for questions!")
            return True

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("🔧 Configuration")
    api_key = st.text_input("Groq API Key", type="password", help="Required for the AI agent")
    model = st.selectbox("LLM Model", ["llama-3.3-70b-versatile", "llama3-8b-8192"], index=0)
    
    st.markdown("---")
    st.subheader("⚙️ Document Settings")
    chunk_size = st.slider("Chunk size", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 50, 400, 100, step=10)
    
    st.subheader("🤖 Agent Settings")
    max_iterations = st.slider("Max iterations", 5, 20, 12, step=2)
    max_time = st.slider("Max time (seconds)", 30, 120, 60, step=15)
    
    # Streaming settings
    st.subheader("💫 UI Settings")
    enable_streaming = st.checkbox("Enable streaming responses", value=True)
    
    if st.session_state.user_name:
        st.markdown("---")
        st.subheader("👤 User Info")
        st.write(f"**Name:** {st.session_state.user_name}")
        if st.button("🔄 Reset Name"):
            st.session_state.user_name = None
            st.rerun()
    
    st.markdown("---")
    st.subheader("📄 PDF Status")
    if st.session_state.get("pdf_name") and st.session_state.get("vectorstore"):
        st.success(f"✅ **{st.session_state.pdf_name}** is ready!")
        st.write("📊 You can ask multiple questions about this PDF")
        st.write("💾 Cached to disk for future sessions")
        if st.button("🗑️ Remove PDF"):
            # Remove from disk too
            if st.session_state.get("pdf_hash"):
                vectorstore_path = get_vectorstore_path(
                    st.session_state.pdf_hash, chunk_size, chunk_overlap
                )
                try:
                    if os.path.exists(vectorstore_path):
                        import shutil
                        shutil.rmtree(vectorstore_path)
                except:
                    pass
            
            for key in ["vectorstore", "cache_key", "pdf_name", "pdf_hash"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    else:
        st.info("📤 No PDF uploaded")
    
    # Export section
    if st.session_state.messages:
        st.markdown("---")
        st.subheader("📥 Export Chat")
        
        col1, col2 = st.columns(2)
        with col1:
            markdown_content = export_chat_as_markdown()
            st.download_button(
                label="📝 Download MD",
                data=markdown_content,
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        
        with col2:
            json_content = export_chat_as_json()
            st.download_button(
                label="📊 Download JSON",
                data=json_content,
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    st.markdown("---")
    if st.button("🗑️ Clear All Data"):
        for key in list(st.session_state.keys()):
            if key.startswith(("vectorstore", "doc_name", "cache_key", "agent", "tools", "llm", "messages", "pdf_name", "pdf_hash")):
                del st.session_state[key]
        st.success("All data cleared!")
        st.rerun()

# ---------- Initialize LLM and Tools ----------
if api_key:
    if "llm" not in st.session_state or st.session_state.llm is None:
        st.session_state.llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            model=model,
            temperature=0.1
        )
    
    tools = [
        Tool(
            name="PDF_Search",
            func=rag_tool_func,
            description="Search and analyze the uploaded PDF document. Use when the user asks questions about: document content, summaries, specific information from the PDF. Input should be the user's question about the PDF."
        ),
        Tool(
            name="Calculator", 
            func=calculator_tool_func,
            description="Perform mathematical calculations. Use when the user asks for arithmetic operations. Input should be a mathematical expression like '15 * 24' or '100 + 50'."
        )
    ]
    
    # Initialize agent using the new API
    if "agent" not in st.session_state or st.session_state.agent is None:
        st.session_state.agent = initialize_agent(
            tools=tools,
            llm=st.session_state.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=max_iterations,
            max_execution_time=max_time
        )

# ---------- MAIN CHAT INTERFACE ----------
if api_key:
    if st.session_state.user_name:
        st.markdown(f"### 💬 Chat with AI Agent - Hello {st.session_state.user_name}! 👋")
    else:
        st.markdown("### 💬 Chat with AI Agent")
        st.info("💡 Tip: Introduce yourself by saying 'My name is [Your Name]' and I'll remember it!")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Example queries
    with st.expander("💡 Try these examples"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📄 Summarize the PDF"):
                st.session_state.example_query = "Please summarize the main content and key points of this PDF document"
        with col2:
            if st.button("🧮 Calculate 15 × 24"):
                st.session_state.example_query = "What is 15 * 24?"
        with col3:
            if st.button("❓ General question"):
                st.session_state.example_query = "What is artificial intelligence?"
    
    # PDF Upload
    st.markdown("### 📎 Upload PDF (Optional)")
    uploaded_file = st.file_uploader(
        "Choose a PDF file to analyze", 
        type=["pdf"], 
        key="pdf_uploader",
        help="Upload a PDF to ask questions about its content"
    )
    
    if uploaded_file is not None:
        if not st.session_state.get("pdf_name") or st.session_state.get("pdf_name") != uploaded_file.name:
            success = process_uploaded_pdf(uploaded_file, chunk_size, chunk_overlap)
    
    # Chat input
    user_input = st.chat_input("Ask me anything...")
    if "example_query" in st.session_state:
        user_input = st.session_state.example_query
        del st.session_state.example_query
    
    if user_input:
        detected_name = extract_name_from_message(user_input)
        if detected_name and not st.session_state.user_name:
            st.session_state.user_name = detected_name
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            # Create placeholder for streaming
            response_placeholder = st.empty()
            
            with st.spinner("🤖 Thinking..."):
                simple_response = handle_simple_query_directly(user_input)
                
                if simple_response:
                    if detected_name and not any(msg.get("content", "").startswith(f"Nice to meet you, {detected_name}") for msg in st.session_state.messages):
                        greeting = f"Nice to meet you, {detected_name}! 👋 "
                        simple_response = greeting + simple_response
                    
                    # Apply streaming effect if enabled
                    if enable_streaming:
                        simulate_streaming_response(simple_response, response_placeholder)
                    else:
                        response_placeholder.write(simple_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": simple_response})
                else:
                    try:
                        convo_context = get_conversation_context(n=3)
                        enhanced_input = f"Previous conversation:\n{convo_context}\n\nUser's current question: {user_input}"
                        
                        # Use the agent with the new API
                        response = st.session_state.agent.invoke({"input": enhanced_input})
                        
                        # Extract the output
                        if isinstance(response, dict):
                            answer = response.get("output", "I'm having trouble processing that request.")
                        else:
                            answer = str(response)
                        
                        if detected_name and not any(msg.get("content", "").startswith(f"Nice to meet you, {detected_name}") for msg in st.session_state.messages):
                            greeting = f"Nice to meet you, {detected_name}! 👋 "
                            answer = greeting + answer
                        
                        # Apply streaming effect if enabled
                        if enable_streaming:
                            simulate_streaming_response(answer, response_placeholder)
                        else:
                            response_placeholder.write(answer)
                        
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                    except Exception as e:
                        error_msg = "I'm having trouble with that request. "
                        if "iteration limit" in str(e).lower():
                            error_msg += "Could you try asking a more specific question?"
                        elif "time limit" in str(e).lower():
                            error_msg += "That's taking too long to process. Could you try a simpler question?"
                        else:
                            error_msg += f"Please try rephrasing your question. (Error: {str(e)})"
                        
                        response_placeholder.write(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

else:
    st.info("🔑 Please provide your Groq API Key in the sidebar to get started!")
    st.markdown("### 🚀 What this AI Agent can do:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📄 PDF Q&A**
        - Upload any PDF above
        - Ask multiple questions about content
        - **Persistent disk storage**
        - Get detailed summaries
        """)
    with col2:
        st.markdown("""
        **🧮 Calculator**  
        - Solve math problems
        - Basic arithmetic
        - Handle expressions
        - Show step-by-step
        """)
    with col3:
        st.markdown("""
        **🤖 Smart Agent**
        - Remembers your name
        - **Context-aware chat history**
        - **Streaming responses**
        - **Export chat transcripts**
        """)
