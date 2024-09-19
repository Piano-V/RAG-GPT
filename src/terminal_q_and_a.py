"""
This module is not part of the main RAG-GPT pipeline and it is only for showing how we can perform RAG using Hugging Face and vectordb in the terminal.

To execute the code, after preparing the python environment and the vector database, in the terminal execute:

python src/terminal_q_and_a.py
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List
from utils.load_config import LoadConfig
from dotenv import load_dotenv
import google.generativeai as genai
import os
import yaml

load_dotenv()

# Initialize GoogleGenerativeAI with Gemini Pro model
genai.configure(
    api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel(
    model_name='gemini-pro')



# Load configuration
APPCFG = LoadConfig()
with open("configs/app_config.yml") as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)
# Load the embedding function
embedding = HuggingFaceEmbeddings(model_name=APPCFG.embedding_model_engine)

# Load the vector database
vectordb = Chroma(persist_directory=APPCFG.persist_directory,
                  embedding_function=embedding)

print("Number of vectors in vectordb:", vectordb._collection.count())

# Prepare the RAG with Google Gemini Pro in terminal
while True:
    question = input("\n\nEnter your question or press 'q' to exit: ")
    if question.lower() == 'q':
        break
    question = "# user new question:\n" + question
    docs = vectordb.similarity_search(question, k=APPCFG.k)
    retrieved_docs_page_content: List[str] = [
        str(x.page_content) + "\n\n" for x in docs]
    retrieved_docs_str = "# Retrieved content:\n\n" + "\n".join(retrieved_docs_page_content)
    prompt = APPCFG.llm_system_role+retrieved_docs_str + "\n\n" + question
    
    print("Prompt for Gemini Pro:")
    print(prompt)  # Print the prompt to debug
    try:
        completion = model.generate_content(prompt,generation_config={
            'temperature': 0,
            'max_output_tokens': 800
        })
        print(completion.text)
    except Exception as e:
        print(f"An Error Occured : ")
