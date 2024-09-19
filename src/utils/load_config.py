import os
from dotenv import load_dotenv
import yaml
from langchain_huggingface import HuggingFaceEmbeddings
from pyprojroot import here
import shutil

load_dotenv()

class LoadConfig:
    def __init__(self) -> None:
        config_path = here("configs/app_config.yml")
        with open(config_path) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # LLM configs
        self.llm_engine = app_config["llm_config"]["engine"]
        self.llm_system_role = app_config["llm_config"]["llm_system_role"]
        self.persist_directory = os.path.join(here(), app_config["directories"]["persist_directory"])
        self.custom_persist_directory = os.path.join(here(), app_config["directories"]["custom_persist_directory"])
        self.embedding_model = HuggingFaceEmbeddings(model_name=app_config["embedding_model_config"]["engine"])

        # Retrieval configs
        self.data_directory = os.path.join(here(), app_config["directories"]["data_directory"])
        self.k = app_config["retrieval_config"]["k"]
        self.embedding_model_engine = app_config["embedding_model_config"]["engine"]
        self.chunk_size = app_config["splitter_config"]["chunk_size"]
        self.chunk_overlap = app_config["splitter_config"]["chunk_overlap"]

        # Summarizer config
        self.max_final_token = app_config["summarizer_config"]["max_final_token"]
        self.token_threshold = app_config["summarizer_config"]["token_threshold"]
        self.summarizer_llm_system_role = app_config["summarizer_config"]["summarizer_llm_system_role"]
        self.character_overlap = app_config["summarizer_config"]["character_overlap"]
        self.final_summarizer_llm_system_role = app_config["summarizer_config"]["final_summarizer_llm_system_role"]
        self.temperature = app_config["llm_config"]["temperature"]

        # Memory
        self.number_of_q_a_pairs = app_config["memory"]["number_of_q_a_pairs"]

        # Clean up the upload doc vectordb if it exists
        self.create_directory(self.persist_directory)
        self.remove_directory(self.custom_persist_directory)

    def create_directory(self, directory_path: str):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def remove_directory(self, directory_path: str):
        if os.path.exists(directory_path):
            try:
                shutil.rmtree(directory_path)
                print(f"The directory '{directory_path}' has been successfully removed.")
            except OSError as e:
                print(f"Error: {e}")
        else:
            print(f"The directory '{directory_path}' does not exist.")
