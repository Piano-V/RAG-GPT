import gradio as gr
import time
import google.generativeai as genai
import os
from langchain_chroma import Chroma
from typing import List, Tuple
from utils.load_config import LoadConfig
from langchain_huggingface import HuggingFaceEmbeddings
import re
import ast
import html

# Load configuration
APPCFG = LoadConfig()

# Initialize Google Generative AI
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel(model_name='gemini-pro')

class ChatBot:
    """
    Class representing a chatbot with document retrieval and response generation capabilities.

    This class provides static methods for responding to user queries, handling feedback, and
    managing chat history.
    """
    @staticmethod
    def respond(chatbot: List, message: str, data_type: str = "Preprocessed doc", temperature: float = 0.0) -> Tuple:
        """
        Generate a response to a user query using document retrieval and language model completion.

        Parameters:
            chatbot (List): List representing the chatbot's conversation history.
            message (str): The user's query.
            data_type (str): Type of data used for document retrieval ("Preprocessed doc" or "Upload doc: Process for RAG").
            temperature (float): Temperature parameter for language model completion.

        Returns:
            Tuple: A tuple containing an empty string, the updated chat history, and references from retrieved documents.
        """
        # Initialize chat session
        chat = model.start_chat()

        if data_type == "Preprocessed doc":
            if os.path.exists(APPCFG.persist_directory):
                embedding = HuggingFaceEmbeddings(model_name=APPCFG.embedding_model_engine)
                vectordb = Chroma(persist_directory=APPCFG.persist_directory,
                                  embedding_function=embedding)
            else:
                chatbot.append(
                    (message, f"VectorDB does not exist. Please first execute the 'upload_data_manually.py' module. For further information please visit {hyperlink}."))
                return "", chatbot, None

        elif data_type == "Upload doc: Process for RAG":
            if os.path.exists(APPCFG.custom_persist_directory):
                vectordb = Chroma(persist_directory=APPCFG.custom_persist_directory,
                                  embedding_function=APPCFG.embedding_model)
            else:
                chatbot.append(
                    (message, f"No file was uploaded. Please first upload your files using the 'upload' button."))
                return "", chatbot, None

        docs = vectordb.similarity_search(message, k=APPCFG.k)
        question = "# User new question:\n" + message

        # Create prompt
        chat_history = "\n".join([f"{item[0]}: {item[1]}" for item in chatbot[-APPCFG.number_of_q_a_pairs:]])
        retrieved_docs_page_content: List[str] = [
        str(x.page_content) + "\n\n" for x in docs]
        retrieved_docs_str = "# Retrieved content:\n\n" + "\n".join(retrieved_docs_page_content)
        prompt = f"{APPCFG.llm_system_role}\nChat history:\n{chat_history}\n\nRetrieved content:\n{retrieved_docs_str}\n\n{question}"

        # Send prompt and get response
        chat.send_message(prompt)
        response = chat.send_message(message)

        # Append response to chat history
        chatbot.append((message, response.text))

        # Return response, updated chat history, and references
        time.sleep(2)
        return "", chatbot, retrieved_docs_str
    

    @staticmethod
    def clean_references(documents: List) -> str:
        """
        Clean and format references from retrieved documents.

        Parameters:
            documents (List): List of retrieved documents.

        Returns:
            str: A string containing cleaned and formatted references.
        """
        server_url = "http://localhost:8000"
        documents = [str(x)+"\n\n" for x in documents]
        markdown_documents = ""
        counter = 1
        for doc in documents:
            # Extract content and metadata
            content, metadata = re.match(
                r"page_content=(.*?)( metadata=\{.*\})", doc).groups()
            metadata = metadata.split('=', 1)[1]
            metadata_dict = ast.literal_eval(metadata)

            # Decode newlines and other escape sequences
            content = bytes(content, "utf-8").decode("unicode_escape")

            # Replace escaped newlines with actual newlines
            content = re.sub(r'\\n', '\n', content)
            # Remove special tokens
            content = re.sub(r'\s*<EOS>\s*<pad>\s*', ' ', content)
            # Remove any remaining multiple spaces
            content = re.sub(r'\s+', ' ', content).strip()

            # Decode HTML entities
            content = html.unescape(content)

            # Replace incorrect unicode characters with correct ones
            content = content.encode('latin1').decode('utf-8', 'ignore')

            # Remove or replace special characters and mathematical symbols
            # This step may need to be customized based on the specific symbols in your documents
            content = re.sub(r'â', '-', content)
            content = re.sub(r'â', '∈', content)
            content = re.sub(r'Ã', '×', content)
            content = re.sub(r'ï¬', 'fi', content)
            content = re.sub(r'â', '∈', content)
            content = re.sub(r'Â·', '·', content)
            content = re.sub(r'ï¬', 'fl', content)

            pdf_url = f"{server_url}/{os.path.basename(metadata_dict['source'])}"

            # Append cleaned content to the markdown string with two newlines between documents
            markdown_documents += f"# Retrieved content {counter}:\n" + content + "\n\n" + \
                f"Source: {os.path.basename(metadata_dict['source'])}" + " | " +\
                f"Page number: {str(metadata_dict['page'])}" + " | " +\
                f"[View PDF]({pdf_url})" "\n\n"
            counter += 1

        return markdown_documents

