import streamlit as st
import os
# import tempfile # Removed tempfile
# from langchain.chains import RetrievalQA # Removed as we are using a custom chain
from langchain.chains import LLMChain # Added for the reasoning and answer chains
from langchain_core.prompts import PromptTemplate # Added for prompt templates
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from operator import itemgetter
import re
from docx import Document
import fitz
from langchain_core.runnables import RunnablePassthrough, RunnableParallel # Added for the parallel and sequential chains
# from google.colab import userdata


# Helper functions for reading different file types
def read_docx(file_content):
    """Reads content from a .docx file."""
    st.write("Reading Documents...")
    try:
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
    st.write("Reading PDFs...")
    try:
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
    st.write("Loading and Pre-processing Transcripts....")
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
@st.cache_resource # Cache the embedding model
def get_embedding_model():
    """Gets the SentenceTransformer embedding model."""
    st.write("Getting Sentence Transformer Embeddings...")
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

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

    embedding_model = get_embedding_model() # Use the cached model
    # Ensure document_chunks is not empty before embedding
    if document_chunks:
        document_embeddings = embedding_model.embed_documents([chunk.page_content for chunk in document_chunks])
    else:
        document_embeddings = []

    return document_chunks, document_embeddings

# Vector store creation function
@st.cache_resource # Cache the vector store creation
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

    st.write("Creating Vector Store...")
    embeddings_model = get_embedding_model() # Use the cached model
    # Using an in-memory vector store for the Streamlit demo
    vectorstore = Chroma.from_documents(document_chunks, embeddings_model)
    return vectorstore

# Streamlit application layout
st.title("BlackBox RAG Pipeline POC for Meeting Transcripts")

st.header("Upload Meeting Transcripts")
uploaded_files = st.file_uploader("Choose transcript files (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], accept_multiple_files=True)

# Define a prompt for the reasoning model to summarize the retrieved documents
# reasoning_prompt_template = """Given the following document chunks, please provide a concise summary that highlights the key information relevant to the user's question.
# Document Chunks:
# {context}

# User Question:
# {question}

# Summary:"""
# st.write("Reasoning Prompt Template Used:  ",reasoning_prompt_template)

# NEW ENHANCED PROMPT
reasoning_prompt_template = """You are an expert meeting analyst AI. Your task is to analyze the provided meeting transcript chunks and generate a structured summary.
Based *only* on the information in the document chunks below, provide the following analysis. If a section contains no relevant information, you MUST explicitly state 'None'.

Document Chunks:
{context}

---

**Meeting Analysis Summary**

**Overall Sentiment:** [Analyze the tone of the discussion - e.g., Positive, Neutral, Negative, Mixed. Provide a brief justification.]
**Meeting Effectiveness:** [Rate the effectiveness based on clear outcomes and decisions vs. unresolved topics. Rate as Effective, Moderately Effective, or Ineffective, and briefly justify your rating.]

### Actions
- [List all concrete action items, assigning owners and deadlines if mentioned.]

### Information
- [Summarize key informational points, updates, and topics that were discussed but did not result in a specific decision or action.]

### Decisions
- [List all firm decisions that were made.]
"""


reasoning_prompt = PromptTemplate(template=reasoning_prompt_template, input_variables=["context", "question"])

# Use a smaller LLM for the reasoning step (e.g., gpt-3.5-turbo or gpt-4o-mini again)
reasoning_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
st.write("Reasoning LLM used: gpt-4o-mini")
# Create a chain for the reasoning model
reasoning_chain = LLMChain(llm=reasoning_llm, prompt=reasoning_prompt, output_key="summary")

# Define a prompt for the main language model to generate the final answer based on the summary
answer_prompt_template = """Given the following summary of relevant information and the user's question, please provide a comprehensive answer.

Summary:
{summary}

User Question:
{question}

Answer:"""
answer_prompt = PromptTemplate(template=answer_prompt_template, input_variables=["summary", "question"])
# st.write("Answer Prompt Template Used:",answer_prompt)

# Use the main LLM for generating the final answer
main_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create a chain for the main language model
answer_chain = LLMChain(llm=main_llm, prompt=answer_prompt, output_key="answer")


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
            # Build RAG chain with reasoning
            retriever = st.session_state.vectorstore.as_retriever()

            # Combine the retriever, reasoning chain, and answer chain into a sequential chain
            # rag_pipeline_with_reasoning = (
            #     RunnableParallel(
            #         {"context": retriever, "question": RunnablePassthrough()}
            #     ) |
            #     RunnableParallel(
            #         {"summary": reasoning_chain, "question": RunnablePassthrough()}
            #     ) |
            #     answer_chain
            # )

            # Create a parallel step to retrieve context and pass the question through
            # CORRECTED CODE
            setup_and_retrieval = RunnableParallel(
                context=itemgetter("question") | retriever,
                question=itemgetter("question"),
            )
            
            # Chain the components sequentially. The output of one step becomes the input to the next.
            rag_pipeline_with_reasoning = (
                setup_and_retrieval
                | reasoning_chain
                | answer_chain
)

            # Check if OPENAI_API_KEY is available (using Streamlit secrets is recommended)
            # os.environ["OPENAI_API_KEY"] = userdata.get("OPEN_API_KEY") # Commented out as per original code
            if "OPENAI_API_KEY" not in os.environ:
                 st.error("OPENAI_API_KEY environment variable not set. Please set it to use the RAG chain.")
            else:
                # try:
                #     # Use the integrated RAG pipeline
                #     answer = rag_pipeline_with_reasoning.invoke({"question": user_question})
                #     st.write("Answer:")
                #     # The output of the chain is a dictionary with 'answer' key
                #     st.info(answer['answer'])
                # except Exception as e:
                #     st.error(f"Error during RAG chain invocation: {e}")
                # NEW DEBUGGING CODE
                try:
                    # 1. Define the chain up to the reasoning step
                    reasoning_step_chain = setup_and_retrieval | reasoning_chain
                
                    # 2. Invoke this partial chain to get the intermediate output
                    intermediate_output = reasoning_step_chain.invoke({"question": user_question})
                    
                    # 3. Display the reasoning model's summary
                    with st.expander("🔍 View Reasoning Model Output"):
                        st.subheader("Intermediate Summary:")
                        st.write(intermediate_output.get("summary", "No summary was generated."))
                        st.subheader("Retrieved Context Sent to Reasoning Model:")
                        st.write(intermediate_output.get("context", "No context was retrieved."))
                
                    # 4. Pass the intermediate output to the final answer chain
                    final_answer = answer_chain.invoke(intermediate_output)
                    
                    st.write("Final Answer:")
                    st.info(final_answer['answer'])
                
                except Exception as e:
                    st.error(f"Error during RAG chain invocation: {e}")


    

    elif user_question and ("vectorstore" not in st.session_state or not st.session_state.vectorstore):
        st.warning("Please upload and process transcripts first.")
else:
    st.info("Please upload meeting transcripts to get started.")
