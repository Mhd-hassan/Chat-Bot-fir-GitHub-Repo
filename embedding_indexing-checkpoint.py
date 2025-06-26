from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document # Import Document class

class EmbeddingIndexer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"): # 
        self.embeddings = HuggingFaceBgeEmbeddings(model_name=model_name)

    def create_and_store_embeddings(self, documents, db_path="faiss_index"):
        # Convert dictionaries to LangChain Document objects
        langchain_documents = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in documents]

        print("Creating and storing embeddings...")
        # Store embeddings in FAISS 
        vector_store = FAISS.from_documents(langchain_documents, self.embeddings)
        vector_store.save_local(db_path)
        print(f"Embeddings stored in {db_path}.")
        return vector_store

    def load_vector_store(self, db_path="faiss_index"):
        print(f"Loading vector store from {db_path}...")
        try:
            vector_store = FAISS.load_local(db_path, self.embeddings, allow_dangerous_deserialization=True)
            print("Vector store loaded successfully.")
            return vector_store
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None

if __name__ == '__main__':
    # Example Usage:
    from repo_handler import RepoHandler

    # Assuming you've run the repo_handler example and have documents
    handler = RepoHandler()
    repo_url = "https://github.com/streamlit/streamlit-example-app"
    repo_dir = handler.clone_repo(repo_url)
    if repo_dir:
        documents = handler.extract_and_chunk_files(repo_dir)
        if documents:
            indexer = EmbeddingIndexer()
            vector_store = indexer.create_and_store_embeddings(documents)
            # You can test loading it back:
            # loaded_vector_store = indexer.load_vector_store()