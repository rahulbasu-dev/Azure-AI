import streamlit as st
import os
# import tempfile # Removed tempfile
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from docx import Document
import fitz
# from google.colab import userdata

# Helper functions for reading different file types
def read_docx(file_content):
    """Reads content from a .docx file."""
    st.write("Reading Documents")
    try:
        # Use BytesIO to read from in-memory bytes
        from io import BytesIO
        doc = Document(BytesIO(file_content))
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return '\n'.join(text)
    except Exception as e:
        st.error(f"Error reading .docx file: {e}")
        return None

def read_pdf(file_content):
    """Reads content from a .pdf file."""
    st.write("Reading PDFs")
    try:
        # Use BytesIO to read from in-memory bytes
        from io import BytesIO
        doc = fitz.open(stream=BytesIO(file_content), filetype="pdf")
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading .pdf file: {e}")
        return None

# Data loading and preprocessing function
def load_and_preprocess_transcripts(uploaded_files):
    """
    Loads and preprocesses meeting transcripts from uploaded Streamlit files.

    Args:
        uploaded_files (list): A list of Streamlit UploadedFile objects.

    Returns:
        list: A list of preprocessed transcript strings.
    """
    st.write("Loading and Pre-processing Transcripts")
    transcripts = []
    for uploaded_file in uploaded_files:
        try:
            file_content = uploaded_file.getvalue()
            if uploaded_file.name.endswith(".txt"):
                transcript_content = file_content.decode('utf-8')
                transcripts.append(transcript_content)
            elif uploaded_file.name.endswith(".docx"):
                transcript_content = read_docx(file_content)
                if transcript_content:
                    transcripts.append(transcript_content)
            elif uploaded_file.name.endswith(".pdf"):
                transcript_content = read_pdf(file_content)
                if transcript_content:
                    transcripts.append(transcript_content)
            else:
                st.warning(f"Skipping unsupported file type: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")


    cleaned_transcripts = []
    for transcript_content in transcripts:
        # Clean and normalize the text
        if transcript_content: # Ensure content is not None from failed reads
            cleaned_content = re.sub(r'[^\w\s.]', '', transcript_content) # Remove characters except alphanumeric, whitespace, and periods
            cleaned_content = cleaned_content.lower() # Convert to lowercase
            cleaned_transcripts.append(cleaned_content)

    return cleaned_transcripts

# Chunking and embedding function
def chunk_and_embed(preprocessed_transcripts):
    """
    Splits and embeds preprocessed transcript strings.

    Args:
        preprocessed_transcripts (list): A list of preprocessed transcript strings.

    Returns:
        tuple: A tuple containing:
            - list: A list of document chunks.
            - list: A list of corresponding embeddings.
    """
    if not preprocessed_transcripts:
        st.warning("No valid transcripts to chunk and embed.")
        return [], []

    st.write("Splitting, Chunking and Embedding Transcript Strings")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    document_chunks = text_splitter.create_documents(preprocessed_transcripts)

    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # Ensure document_chunks is not empty before embedding
    if document_chunks:
        document_embeddings = embedding_model.embed_documents([chunk.page_content for chunk in document_chunks])
    else:
        document_embeddings = []

    return document_chunks, document_embeddings

# Vector store creation function
def create_vector_store(document_chunks, document_embeddings):
    """
    Creates a Chroma vector store from document chunks and embeddings.

    Args:
        document_chunks (list): A list of document chunks.
        document_embeddings (list): A list of corresponding embeddings.

    Returns:
        Chroma: The created vector store object, or None if no chunks.
    """
    if not document_chunks:
        st.warning("No document chunks to create vector store.")
        return None

    st.write("Creating Vector Store")
    embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # Using an in-memory vector store for the Streamlit demo
    vectorstore = Chroma.from_documents(document_chunks, embeddings_model)
    return vectorstore

# Streamlit application layout
st.title("BB RAG Pipeline for Meeting Transcripts")

st.header("Upload Meeting Transcripts")
uploaded_files = st.file_uploader("Choose transcript files (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    # Process uploaded files only if different from previous uploads
    if "uploaded_file_names" not in st.session_state or st.session_state.uploaded_file_names != [f.name for f in uploaded_files]:
        st.session_state.uploaded_file_names = [f.name for f in uploaded_files]
        with st.spinner("Processing transcripts..."):
            preprocessed_data = load_and_preprocess_transcripts(uploaded_files)
            document_chunks, document_embeddings = chunk_and_embed(preprocessed_data)
            st.session_state.vectorstore = create_vector_store(document_chunks, document_embeddings)
        if "vectorstore" in st.session_state and st.session_state.vectorstore:
            st.success("Transcripts processed and vector store created!")
        else:
            st.error("Failed to process transcripts and create vector store. Please check the uploaded files.")


    st.header("Ask a Question")
    user_question = st.text_input("Enter your question:")

    if user_question and "vectorstore" in st.session_state and st.session_state.vectorstore:
        with st.spinner("Getting answer..."):
            # Build RAG chain (re-instantiate LLM and retriever as needed by Streamlit's execution model)
            # Assuming OPENAI_API_KEY is set as an environment variable
            # os.environ["OPENAI_API_KEY"] = userdata.get("OPEN_API_KEY") # Commented out as per original code

            # Check if OPENAI_API_KEY is available
            if "OPENAI_API_KEY" not in os.environ:
                 st.error("OPENAI_API_KEY environment variable not set. Please set it to use the RAG chain.")
            else:
                try:
                    llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"],model="gpt-4o-mini", temperature=0)
                    retriever = st.session_state.vectorstore.as_retriever()
                    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

                    answer = rag_chain.invoke({"query": user_question})
                    st.write("Answer:")
                    st.info(answer['result'])
                except Exception as e:
                    st.error(f"Error during RAG chain invocation: {e}")

    elif user_question and ("vectorstore" not in st.session_state or not st.session_state.vectorstore):
        st.warning("Please upload and process transcripts first.")
else:
    st.info("Please upload meeting transcripts to get started.")
