import os
import shutil
import time
import uuid
from git import Repo
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Add your Hugging Face Hub token here
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

class RepoHandler:
    def __init__(self, local_path="cloned_repos"):
        self.local_path = local_path
        os.makedirs(self.local_path, exist_ok=True)

    def _safe_remove_directory(self, directory):
        retries = 3
        last_error = None
        for i in range(retries):
            try:
                if os.path.exists(directory):
                    # First, ensure all files are writable
                    for root, dirs, files in os.walk(directory):
                        for dir_name in dirs:
                            dir_path = os.path.join(root, dir_name)
                            os.chmod(dir_path, 0o777)
                        for file_name in files:
                            file_path = os.path.join(root, file_name)
                            os.chmod(file_path, 0o777)

                    def onerror(func, path, exc_info):
                        import stat
                        try:
                            os.chmod(path, stat.S_IWRITE)
                            func(path)
                        except Exception as e:
                            nonlocal last_error
                            last_error = str(e)

                    shutil.rmtree(directory, ignore_errors=False, onerror=onerror)
                    time.sleep(0.5)  # Small delay to ensure file handles are released

                    if not os.path.exists(directory):
                        return True, None
            except Exception as e:
                last_error = str(e)
                print(f"Attempt {i+1} failed to remove directory: {e}")
                if i < retries - 1:
                    time.sleep(1)  # Wait before retry
        return False, last_error or "Unknown error removing directory."

    def clone_repo(self, repo_url):
        repo_name = repo_url.split('/')[-1].replace(".git", "")
        # Use a unique directory for each clone to avoid deletion issues
        unique_id = str(uuid.uuid4())[:8]
        repo_dir = os.path.join(self.local_path, f"{repo_name}_{unique_id}")

        try:
            print(f"Cloning {repo_url} into {repo_dir}...")
            Repo.clone_from(repo_url, repo_dir)
            print("Cloning complete.")
            return repo_dir, None
        except Exception as e:
            print(f"Error cloning repository: {e}")
            return None, str(e)

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
    repo_url = "https://github.com/Mhd-hassan/Assignment-1.git"
    repo_dir, error = handler.clone_repo(repo_url)
    if repo_dir:
        documents = handler.extract_and_chunk_files(repo_dir)
        print(f"Extracted {len(documents)} chunks from the repository.")
        # print(documents[0]) # Uncomment to see an example chunk
    else:
        print(f"Failed to clone repository: {error}")