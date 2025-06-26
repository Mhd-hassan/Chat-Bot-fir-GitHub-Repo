import streamlit as st
import os
import shutil

from repo_handler import RepoHandler
from embedding_indexing import EmbeddingIndexer
from chat_agent import ChatAgent

# Initialize session state for storing chat history and vector store
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "repo_dir" not in st.session_state:
    st.session_state.repo_dir = None

st.set_page_config(page_title="Chat with GitHub Agent", layout="wide")
st.title("Chat with GitHub Repository AI Agent")

# Sidebar for controls
with st.sidebar:
    st.header("Repository Configuration")
    repo_url_input = st.text_input("Enter GitHub Repository URL:", key="repo_url_input")
    load_repo_button = st.button("Clone & Load Repo", key="load_repo_button")

    if load_repo_button and repo_url_input:
        with st.spinner("Cloning and processing repository... This may take a while."):
            repo_handler = RepoHandler()
            if st.session_state.repo_dir and os.path.exists(st.session_state.repo_dir):
                shutil.rmtree(st.session_state.repo_dir) # Clean up previous repo
                st.session_state.repo_dir = None

            repo_dir = repo_handler.clone_repo(repo_url_input)
            if repo_dir:
                st.session_state.repo_dir = repo_dir # Store the repo directory in session state
                documents = repo_handler.extract_and_chunk_files(repo_dir)
                if documents:
                    indexer = EmbeddingIndexer()
                    st.session_state.vector_store = indexer.create_and_store_embeddings(documents)
                    st.success("Repository loaded and indexed successfully! You can now start chatting.")
                    st.session_state.chat_history = [] # Clear chat history for new repo
                else:
                    st.error("No relevant files found or extracted from the repository.")
            else:
                st.error("Failed to clone the repository. Please check the URL and ensure it's public.")
    elif load_repo_button and not repo_url_input:
        st.warning("Please enter a GitHub Repository URL.")

# Main chat interface
st.subheader("Chat with the Repository")

# Display chat history 
for i, message in enumerate(st.session_state.chat_history):
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**AI:** {message['content']}")
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- File: `{source.metadata.get('file_path')}` (Language: `{source.metadata.get('language')}`)")
                    # st.markdown(f"  Snippet: ```\n{source.page_content[:200]}...\n```") # Uncomment to see snippets


user_query = st.text_input("Ask a question about the repository:", key="user_query_input")
send_button = st.button("Send", key="send_button")

if send_button and user_query:
    if st.session_state.vector_store:
        with st.spinner("Generating response..."):
            agent = ChatAgent(st.session_state.vector_store)
            ai_response, source_documents = agent.chat(user_query)

            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.session_state.chat_history.append({"role": "ai", "content": ai_response, "sources": source_documents})
            st.rerun() # Rerun to update chat history immediately
    else:
        st.warning("Please load a GitHub repository first using the sidebar.")
elif send_button and not user_query:
    st.warning("Please enter a question to chat.")

# Cleanup cloned repository when application exits or tab is closed
# This part is tricky with Streamlit's lifecycle. A more robust solution for
# cleanup might involve background tasks or periodic cleanup for deployed apps.
# For local testing, you might manually delete the 'cloned_repos' directory.
# This simple approach works if Streamlit terminates cleanly.
@st.cache_resource(experimental_allow_widgets=True)
def cleanup_on_exit():
    if st.session_state.repo_dir and os.path.exists(st.session_state.repo_dir):
        print(f"Cleaning up {st.session_state.repo_dir}")
        shutil.rmtree(st.session_state.repo_dir)

# Register the cleanup function (might not always execute reliably on unexpected exits)
# import atexit
# atexit.register(cleanup_on_exit)