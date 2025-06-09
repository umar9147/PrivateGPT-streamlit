import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
)
import streamlit as st
import tempfile

# ====== CONFIGURATION ======
API_KEY = API_KEY = st.secrets["API_KEY"]
genai.configure(api_key=API_KEY)

# ====== STREAMLIT PAGE CONFIG ======
st.set_page_config(
    page_title="PrivateGPT - Document Q&A",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover { background-color: #45a049; }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
    }
    .chat-message { padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem; display: flex; flex-direction: column; }
    .chat-message.user { background-color: #e3f2fd; color: #000000; }
    .chat-message.assistant { background-color: #f5f5f5; color: #000000; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 PrivateGPT - Document Q&A System")
st.markdown("""
    Upload your documents and ask questions about their content. The system will provide answers based on the uploaded documents.<br>
    <b>Supported file types:</b> PDF, TXT, DOCX, PPTX
""", unsafe_allow_html=True)

# ====== SESSION STATE INIT ======
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ====== FILE UPLOAD SECTION ======
st.markdown("### 📄 Upload Documents")
uploaded_files = st.file_uploader(
    "Choose your files",
    type=['pdf', 'txt', 'docx', 'pptx'],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing documents..."):
        all_docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            try:
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                if ext == ".txt":
                    loader = TextLoader(tmp_path)
                elif ext == ".pdf":
                    loader = PyPDFLoader(tmp_path)
                elif ext == ".docx":
                    loader = Docx2txtLoader(tmp_path)
                elif ext == ".pptx":
                    loader = UnstructuredPowerPointLoader(tmp_path)
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue
                docs = loader.load()
                all_docs.extend(docs)
                if uploaded_file.name not in st.session_state.documents:
                    st.session_state.documents.append(uploaded_file.name)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            finally:
                os.unlink(tmp_path)
        if all_docs:
            embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=API_KEY
            )
            st.session_state.vectorstore = FAISS.from_documents(all_docs, embedding_model)
            st.success(f"Successfully processed {len(st.session_state.documents)} document(s)")

# ====== DISPLAY UPLOADED DOCUMENTS ======
if st.session_state.documents:
    st.markdown("### 📚 Uploaded Documents")
    for doc in st.session_state.documents:
        st.markdown(f"- {doc}")

# ====== QUESTION/ANSWER SECTION ======
def strict_rag_answer(question, vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    relevant_docs = retriever.get_relevant_documents(question)
    if not relevant_docs or all(len(doc.page_content.strip()) < 10 for doc in relevant_docs):
        return "❌ Your question is not related to the provided documents."
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    relevance_check_prompt = f"""
Based on the following documents and question, determine if the documents contain relevant information to answer the question.
Answer with only 'YES' or 'NO'.

Documents:
{context}

Question: {question}
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    relevance_response = model.generate_content(relevance_check_prompt).text.strip().upper()
    if relevance_response != "YES":
        return "❌ Your question is not related to the provided documents. Please ask a question about the content in the documents."
    # Strict prompt: only answer from context, otherwise say "I don't know."
    prompt = f"""
You are PrivateGPT, a helpful assistant.
Answer the user's question using  the most relevant paragraphs or sections from the provided documents.
If the answer is not explicitly stated, reply with "I don't know."

Documents:
{context}

Question: {question}
Answer:
"""
    answer = model.generate_content(prompt).text.strip()
    if "i don't know" in answer.lower():
        return "❌ Your question is not related to the provided documents. Please ask a question about the content in the documents."
    return answer

if st.session_state.vectorstore:
    st.markdown("### 💭 Ask Questions")
    question = st.text_input("Enter your question about the documents:")
    if question:
        if st.button("Get Answer"):
            with st.spinner("Thinking..."):
                response = strict_rag_answer(question, st.session_state.vectorstore)
                st.session_state.chat_history.append({"role": "user", "content": question})
                st.session_state.chat_history.append({"role": "assistant", "content": response})

# ====== DISPLAY CHAT HISTORY ======
if st.session_state.chat_history:
    st.markdown("### 💬 Chat History")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user">
                    <strong>You:</strong><br>
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message assistant">
                    <strong>Assistant:</strong><br>
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)