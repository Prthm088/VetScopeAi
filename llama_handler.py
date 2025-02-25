import time
import os
os.environ["REPLICATE_API_TOKEN"] = "r8_Sv1cWIMHvRSbB2QjQ1qTQQGEbWvCOii4G4XFo"
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.replicate import Replicate
from transformers import AutoTokenizer
# set the LLM
llama2_7b_chat = "meta/meta-llama-3.1-405b-instruct"
Settings.llm = Replicate(
    model=llama2_7b_chat,
    temperature=0.01,
    additional_kwargs={"top_p": 1, "max_new_tokens": 300},
)
# set tokenizer to match LLM
Settings.tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf"
)
# set the embed model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
from llama_index.core import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="data")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()

def chatbot_response(que):
    try:
        time.sleep(5)
        answer = query_engine.query(que)
        
    except:
        answer="Sorry, Something Went Wrong!"
        return answer
    return answer
     