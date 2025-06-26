import os
import shutil
from git import Repo
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RepoHandler:
    def __init__(self, local_path="cloned_repos"):
        self.local_path = local_path
        os.makedirs(self.local_path, exist_ok=True)

    def clone_repo(self, repo_url):
        repo_name = repo_url.split('/')[-1].replace(".git", "")
        repo_dir = os.path.join(self.local_path, repo_name)

        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir) # Clear existing repo to ensure fresh clone

        try:
            print(f"Cloning {repo_url} into {repo_dir}...")
            Repo.clone_from(repo_url, repo_dir)
            print("Cloning complete.")
            return repo_dir
        except Exception as e:
            print(f"Error cloning repository: {e}")
            return None

    def extract_and_chunk_files(self, repo_dir):
        raw_documents = []
        for root, _, files in os.walk(repo_dir):
            for file in files:
                # Extract specific file types as mentioned in the PDF 
                if file.endswith(('.py', '.js', '.ts', '.md', '.txt', '.json', '.xml', '.html', '.css', '.java', '.c', '.cpp', '.h', '.hpp', '.go', '.rb', '.php')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        # Storing metadata per chunk as suggested 
                        raw_documents.append({
                            "content": content,
                            "metadata": {
                                "file_name": file,
                                "file_path": os.path.relpath(file_path, repo_dir),
                                "language": file.split('.')[-1], # Simple language detection
                                "source": file_path
                            }
                        })
                    except Exception as e:
                        print(f"Could not read file {file_path}: {e}")

        # Chunk code into small pieces (e.g., function-level or 100-300 token chunks) 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # Using a larger chunk size initially, can be tuned
            chunk_overlap=200
        )

        # Process each document to create LangChain Document objects
        documents = []
        for doc_info in raw_documents:
            chunks = text_splitter.split_text(doc_info["content"])
            for i, chunk in enumerate(chunks):
                metadata = doc_info["metadata"].copy()
                metadata["chunk_number"] = i
                # Note: LangChain's text splitter doesn't directly give line numbers
                # Tree-sitter (optional) would be needed for precise function-level parsing and line numbers
                documents.append(
                    {"page_content": chunk, "metadata": metadata}
                )
        return documents

if __name__ == '__main__':
    # Example Usage:
    handler = RepoHandler()
    # Replace with a public GitHub repo URL for testing
    repo_url = "https://github.com/streamlit/streamlit-example-app"
    repo_dir = handler.clone_repo(repo_url)
    if repo_dir:
        documents = handler.extract_and_chunk_files(repo_dir)
        print(f"Extracted {len(documents)} chunks from the repository.")
        # print(documents[0]) # Uncomment to see an example chunk