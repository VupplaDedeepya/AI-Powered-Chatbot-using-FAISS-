import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
import os

# --- HTML/CSS for Chat UI ---
# This is equivalent to the contents of htmlTemplates.py
css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="data:image/jpeg;base64,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">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

# --- Core Logic Functions ---

# Extracts text from a list of PDF files.
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            continue
    return text

# Splits raw text into smaller, manageable chunks.
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, # Increased chunk size for better context
        chunk_overlap=200, # Increased overlap to prevent loss of information
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Creates a vector store from text chunks using HuggingFace embeddings.
def get_vector_store(text_chunks):
    if not text_chunks:
        st.warning("Cannot create vector store: No text chunks available.")
        return None
    
    # Using a robust embedding model
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    try:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Creates a conversational retrieval chain using a reliable LLM.
def get_conversation_chain(vectorstore):
    if not vectorstore:
        return None
    
    # Check for the HuggingFace API key
    if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
        st.error("HUGGINGFACEHUB_API_TOKEN environment variable not found. Please add it to a .env file or your secrets.")
        return None

    # Use HuggingFacePipeline for a more stable integration with a smaller model
    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base", # Changed to a smaller model to prevent storage issues
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Handles user questions and updates the chat history.
def handle_userinput(user_question):
    if not st.session_state.conversation:
        st.warning("Please upload and process a document first.")
        return

    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
    except Exception as e:
        st.error(f"Error processing your question: {e}")

# Displays the chat history on the page.
def display_chat_history():
    if st.session_state.chat_history:
        # Iterate in reverse to show the latest messages at the bottom
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# Main function to run the Streamlit app.
def main():
    load_dotenv()
    st.set_page_config(page_title="AI BOT", page_icon=":robot_face:", layout='wide')
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Welcome to AI BOT :robot_face:")
    st.write("Upload a document on the left, then ask me questions about it!")

    # Display the current chat history
    display_chat_history()

    # User input text box
    user_question = st.text_input("Ask me a question about your Document:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar for document upload and processing
    with st.sidebar:
        st.subheader("Your Document")
        pdf_docs = st.file_uploader("Upload your document here", type="pdf", accept_multiple_files=True)
        
        if st.button("Process Document"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF document.")
                return

            with st.spinner("Processing..."):
                # Extract text from PDFs
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text:
                    st.warning("No text could be extracted from the uploaded PDFs.")
                    return

                # Split text into chunks
                text_chunks = get_text_chunks(raw_text)
                if not text_chunks:
                    st.warning("No text chunks could be created from the document.")
                    return

                # Create vector store
                vectorstore = get_vector_store(text_chunks)
                if vectorstore:
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    if st.session_state.conversation:
                        st.success("Document processed and conversation chain is ready! You can now ask questions.")
                    else:
                        st.error("Failed to create conversation chain. Check API key and model availability.")


if __name__ == '__main__':
    main()
