from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
from IPython.display import Markdown, display
import chromadb

from pathlib import Path
from llama_index import download_loader

PDFReader = download_loader("PDFReader")

loader = PDFReader()
documents = loader.load_data(file=Path('/content/ARM-RAG.pdf'))

# huggingface api token for downloading llama2
hf_token = "hf_token"

import torch
from transformers import BitsAndBytesConfig
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    query_wrapper_prompt=PromptTemplate("<s> [INST] {query_str} [/INST] "),
    context_window=3900,
    model_kwargs={"token": hf_token, "quantization_config": quantization_config},
    tokenizer_kwargs={"token": hf_token},
    device_map="auto",
)


resp = llm.complete("What is ARM-RAG? ")

from IPython.display import HTML, display

# Assuming resp contains the response
resp = llm.complete("What is ARM-RAG?")

# Using HTML with inline CSS for styling (gray color, smaller font size)
html_text = f'<p style="color: #1f77b4; font-size: 14px;"><b>{resp}</b></p>'
display(HTML(html_text))

# create client and a new collection
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

# define embedding function
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, service_context=service_context

from llama_index import ServiceContext

service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")


from llama_index import VectorStoreIndex

vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context)

from llama_index import SummaryIndex

summary_index = SummaryIndex.from_documents(documents, service_context=service_context)

from llama_index.response.notebook_utils import display_response

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

query_engine = vector_index.as_query_engine(response_mode="compact")

response = query_engine.query("What is ARM-RAG?")

display_response(response)

query="What machine learning techniques does ARM-RAG leverage?"

from llama_index.embeddings.base import similarity
query_engine =index.as_query_engine(response_mode="tree_summarize")

response = query_engine.query(query)

from IPython.display import HTML, display

# Using HTML with inline CSS for styling (blue color)
html_text = f'<p style="color: #1f77b4; font-size: 14px;"><b>{response}</b></p>'
display(HTML(html_text))

from llama_index.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        "You are a chatbot, able to have normal interactions"
        " about ARM-RAG"
    ),
)


response = chat_engine.chat("What machine learning techniques does ARM-RAG leverage?")
print(response)

response= chat_engine.chat("give me real world examples of apps/system i can build leveraging ARM-RAG?")

print(response)

response = chat_engine.chat("so ARM-RAG is just all about explanations in real world application?")

print(response)

