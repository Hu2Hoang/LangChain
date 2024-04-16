from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from pprint import pprint
import torch
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
import time
start = time.time()
model_file = "models/PhoGPT-4B-Chat-Q8_0.gguf"
vector_db_path = "vectorstores/db_faiss"

def load_llm(model_file):
  
  n_gpu_layers = 20
  n_batch =512
  callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
  # Load the Llama model with llama-cpp-python
  llm = LlamaCpp(
    model_path=model_file,
    temperature=1,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    max_tokens=4096,
    callback_manager=callback_manager,
    verbose=True,
    n_ctx=2048,
    )

  return llm

def creat_prompt(template):
    prompt = PromptTemplate(template = PROMPT_TEMPLATE, input_variables=["context", "question"])
    return prompt


def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":3}, max_tokens_limit=2048),
        return_source_documents = True,
        chain_type_kwargs= {'prompt': prompt}

    )
    return llm_chain

def read_vectors_db():
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(vector_db_path, embedding_model,allow_dangerous_deserialization=True)
    return db


db = read_vectors_db()
llm = load_llm(model_file)
PROMPT_TEMPLATE = "### Câu hỏi: Dựa vào thông tin sau: \n{context}\nHãy trả lời đúng, đủ câu hỏi, nếu không biết thì trả lời không biết: {question}\n ### Trả lời:"  

prompt = creat_prompt(PROMPT_TEMPLATE)
# Callbacks support token-wise streaming
llm_chain  = create_qa_chain(prompt, llm, db)


# run chain
question = "Trần Anh Tuấn trong MSB có vai trò gì?"
response = llm_chain.invoke({"query": question})
pprint(response)

end = time.time()
print("The time of execution of above program is :",
      (end-start) * 10**3, "ms")