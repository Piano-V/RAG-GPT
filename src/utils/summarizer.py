from langchain_community.document_loaders import PyPDFLoader
import google.generativeai as genai
import os

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel(model_name='gemini-pro')

class Summarizer:
    @staticmethod
    def summarize_the_pdf(
        file_dir: str,
        max_final_token: int,
        token_threshold: int,
        gpt_model: str,
        temperature: float,
        summarizer_llm_system_role: str,
        final_summarizer_llm_system_role: str,
        character_overlap: int
    ) -> str:
        """
        Summarizes the content of a PDF file using Gemini Pro's engine.

        Args:
            file_dir (str): The path to the PDF file.
            max_final_token (int): The maximum number of tokens in the final summary.
            token_threshold (int): The threshold for token count reduction.
            gpt_model (str): The ChatGPT engine model name.
            temperature (float): The temperature parameter for ChatGPT response generation.
            summarizer_llm_system_role (str): The system role for the summarizer.
            final_summarizer_llm_system_role (str): The final summarizer system role.
            character_overlap (int): The number of overlapping characters between pages.

        Returns:
            str: The final summarized content.
        """
        docs = PyPDFLoader(file_dir).load()
        print(f"Document length: {len(docs)}")
        max_summarizer_output_token = int(max_final_token / len(docs)) - token_threshold
        full_summary = ""
        counter = 1

        print("Generating the summary..")
        if len(docs) > 1:
            for i in range(len(docs)):
                if i == 0:  # First page
                    prompt = docs[i].page_content + docs[i+1].page_content[:character_overlap]
                elif i < len(docs) - 1:  # Middle pages
                    prompt = docs[i-1].page_content[-character_overlap:] + docs[i].page_content + docs[i+1].page_content[:character_overlap]
                else:  # Last page
                    prompt = docs[i-1].page_content[-character_overlap:] + docs[i].page_content

                summarizer_prompt = summarizer_llm_system_role.format(max_summarizer_output_token) + prompt
                try:
                    response = model.generate_content(summarizer_prompt, generation_config={
                        'temperature': temperature
                    })
                    page_summary = response.text
                    full_summary += page_summary
                except Exception as e:
                    print(f"Error summarizing page {counter}: {e}")

                print(f"Page {counter} was summarized. ", end="")
                counter += 1
        else:
            full_summary = docs[0].page_content
            print(f"Page {counter} was summarized. ", end="")
            counter += 1

        print("\nFull summary token length is not calculated due to tokenizer issues.")
        final_prompt = final_summarizer_llm_system_role + full_summary
        try:
            final_response = model.generate_content(final_prompt, generation_config={
                'temperature': temperature
            })
            return final_response.text
        except Exception as e:
            print(f"Error generating final summary: {e}")
            return "Error generating final summary."

