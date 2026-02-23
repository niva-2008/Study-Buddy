import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# --- 2026 Modular Imports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- Compatibility Imports ---
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# 1. Load API Key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# 2. UI Configuration
st.set_page_config(page_title="AI Study Buddy", page_icon="🎓", layout="wide")

# Initialize persistent memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.title("📂 Study Materials")
    pdfs = st.file_uploader("Upload PDF notes", type="pdf", accept_multiple_files=True)
    process_clicked = st.button("Analyze Notes")
    
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    model_choice = st.selectbox("LLM Brain", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"])

# --- Logic: PDF to Vector Store ---
if process_clicked and pdfs:
    with st.spinner("Buddy is reading your notes..."):
        all_text = ""
        for pdf in pdfs:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text_content = page.extract_text()
                if text_content: all_text += text_content
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(all_text)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embeddings)
        st.session_state.retriever = vector_store.as_retriever()
        st.session_state.is_ready = True
        st.success("I've finished reading! Ask me anything.")

# --- Main Chat Interface ---
st.title("🎓 AI Study Buddy")

# Display Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if "is_ready" in st.session_state:
    if user_input := st.chat_input("Ask a doubt, request a quiz, or ask for answers..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_choice)
                
                # Create a string of the last few messages so the AI has context
                history_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:]])
                
                input_lower = user_input.lower()
                
                # --- DYNAMIC INSTRUCTION LOGIC ---
                if any(word in input_lower for word in ["quiz", "test", "examine"]):
                    mode_instruction = """
                    MODE: QUIZ MASTER.
                    1. Generate exactly 5 quiz questions based on the context.
                    2. Use of Multiple Choice (A-D) .
                    3. Format MCQs line-by-line.
                    4. IMPORTANT: Do NOT provide answers in this response.
                    """
                elif any(word in input_lower for word in ["answer", "solution", "reveal"]):
                    mode_instruction = f"""
                    MODE: ANSWER KEY.
                    Look at the questions you just generated in the chat history:
                    {history_context}
                    Provide the correct answers for those specific questions using the context.
                    """
                else:
                    mode_instruction = """
                    MODE: TUTOR.
                    Answer the student's doubt clearly and cleanly using the provided context. 
                    Use bullet points for key facts.
                    """

                # Note the double {{ }} for context/input so Python f-strings don't break
                prompt_template = ChatPromptTemplate.from_template(f"""
                System Instruction: {mode_instruction}
                
                Context: {{context}}
                
                Student Question: {{input}}
                """)

                # Construct RAG Chain
                doc_chain = create_stuff_documents_chain(llm, prompt_template)
                rag_chain = create_retrieval_chain(st.session_state.retriever, doc_chain)
                
                # Execute
                result = rag_chain.invoke({"input": user_input})
                answer = result["answer"]
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👋 Upload your PDF notes to begin!")