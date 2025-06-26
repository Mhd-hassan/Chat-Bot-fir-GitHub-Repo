from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline # Use HuggingFacePipeline for local models
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch # Import torch for device mapping

class ChatAgent:
    def __init__(self, vector_store):
        self.vector_store = vector_store

        # --- Local LLM Setup (Alternative to HuggingFaceHub) ---
        model_name = "mistralai/Mistral-7B-Instruct-v0.2" # Or a smaller model like "distilbert/distilgpt2" for testing
        
        # Load tokenizer and model
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Check for CUDA availability and set device map accordingly
            device_map = "auto"
            if not torch.cuda.is_available():
                print("CUDA not available, loading model on CPU. This will be very slow.")
                device_map = {"": "cpu"} # Explicitly map to CPU if no GPU

            # Load model with bfloat16 for efficiency if GPU is available, otherwise default
            # Use load_in_8bit or load_in_4bit with bitsandbytes for less memory if GPU
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, # Use bfloat16/float16 for GPU, float32 for CPU
                # load_in_8bit=True, # Uncomment this if you have a GPU and want to save memory
                # quantization_config=BitsAndBytesConfig(load_in_4bit=True), # For 4-bit quantization
            )
            
            # Create a Hugging Face pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
                top_k=50,
                repetition_penalty=1.1,
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)

        except Exception as e:
            print(f"Error loading local LLM: {e}")
            print("Falling back to HuggingFaceHub (requires HUGGINGFACEHUB_API_TOKEN environment variable).")
            # Fallback to HuggingFaceHub if local loading fails
            llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
            from langchain.llms import HuggingFaceHub # Import here to avoid circular dependency if not needed
            self.llm = HuggingFaceHub(
                repo_id=llm_model,
                model_kwargs={"temperature": 0.1, "max_length": 512}
            )
        # --- End Local LLM Setup ---

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        template = """You are an AI assistant helping users understand and navigate GitHub repositories.
        Use the following context to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        You should also refer to the filename and line numbers where possible for code-related questions.

        Chat History:
        {chat_history}

        Context:
        {context}

        Question: {question}
        Helpful Answer:"""

        self.qa_prompt = PromptTemplate(
            input_variables=["chat_history", "context", "question"],
            template=template
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.qa_prompt}
        )

    def chat(self, query):
        print(f"User query: {query}")
        response = self.qa_chain({"query": query})
        return response["result"], response["source_documents"]

# Rest of the if __name__ == '__main__': block remains the same