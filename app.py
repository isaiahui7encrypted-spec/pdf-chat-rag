import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="🤖"
)
st.header("Chat with your PDF (RAG System) 📄")

# Initialize chat history in session state
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "input_key" not in st.session_state:
    st.session_state.input_key = 0
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

def get_pdf_text(pdf_docs):
    text_by_file = {}
    for pdf in pdf_docs:
        text = ""
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
            text_by_file[pdf.name] = text
        except Exception as e:
            st.warning(f"Could not read file: {pdf.name} — {e}")
    return text_by_file

def get_text_chunks(text_by_file):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks_with_meta = []
    for filename, text in text_by_file.items():
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            chunks_with_meta.append({"text": chunk, "source": filename})
    return chunks_with_meta

def get_vector_store(text_chunks_with_meta):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    texts = [item["text"] for item in text_chunks_with_meta]
    metadatas = [{"source": item["source"]} for item in text_chunks_with_meta]

    vector_store = FAISS.from_texts(
        texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in provided context just say, "The answer is not
    available in the document", don't provide the wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.3
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = prompt | model
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    if not os.path.exists("faiss_index/index.faiss"):
        st.warning("Please upload and process a PDF file first!")
        return

    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question, k=1)
    context = "\n\n".join([doc.page_content for doc in docs])

    chain = get_conversational_chain()
    response = chain.invoke({"context": context, "question": user_question})

    # Extract source filenames
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", None)
        if source and source not in sources:
            sources.append(source)

    # Save to chat history
    st.session_state.chat_history.append({
        "question": user_question,
        "answer": response.content,
        "sources": sources
    })
# UI — Sidebar
with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)

    if pdf_docs:
        st.markdown("**📂 Uploaded Files:**")
        for pdf in pdf_docs:
            st.write(f"✅ {pdf.name}")

    if st.button("Submit & Process"):
        if not pdf_docs:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Processing..."):
                text_by_file = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(text_by_file)
                get_vector_store(text_chunks)
                st.session_state.pdf_processed = True  # mark as processed
                st.success(f"Done! Processed {len(pdf_docs)} file(s)")
    # Clear Chat always visible
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.pdf_processed = False
        st.rerun()
# Chat input
pdf_processed = os.path.exists("faiss_index/index.faiss")

user_question = st.text_input(
    "Ask a Question from the PDF Files",
    key=f"input_{st.session_state.input_key}",
    disabled=not st.session_state.pdf_processed,
    placeholder="⬅️ Upload and process a PDF file first..." if not st.session_state.pdf_processed else "Ask a question here..."
)

if user_question and user_question != st.session_state.last_question:
    st.session_state.last_question = user_question
    with st.spinner("Thinking..."):
        user_input(user_question)
        st.session_state.input_key += 1
        st.rerun()

# Display chat history (most recent first)
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 Chat History")
    for chat in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            # Show sources below answer
            if chat["sources"]:
                st.markdown("📁 **Found in:**")
                for source in chat["sources"]:
                    st.markdown(f"- 📄 `{source}`")
            else:
                st.markdown("*📁 Source: Unknown*")