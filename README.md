# RAG-GPT: Retrieval Augmented generation (RAG) chatbot using OpenAI GPT Model, Langchain, ChromaDB, and Gradio

**RAG-GPT** supports both PDFs and Docs. 

**The chatbot offers versatile usage through three different methods:**
1. **Offline Documents**: Engage with documents that you've pre-processed and vectorized. These documents can be seamlessly integrated into your chat sessions.
2. **Real-time Uploads:** Easily upload documents during your chat sessions, allowing the chatbot to process and respond to the content on-the-fly.
3. **Summarization Requests:** Request the chatbot to provide a comprehensive summary of an entire PDF or document in a single interaction, streamlining information retrieval.

To employ any of these methods, simply configure the appropriate settings in the "RAG with" dropdown menu within the chatbot interface. Tailor your interactions with documents to suit your preferences and needs efficiently.

* The project provides guidance on configuring various settings, such as adjusting the GPT model's temperature for optimal performance.
* The user interface is crafted with gradio, ensuring an intuitive and user-friendly experience.
* The model incorporates memory, retaining user Q&As for an enhanced and personalized user experience.
* For each response, you can access the retrieved content along with the option to view the corresponding PDF. 

## RAG-GPT User Interface
<div align="center">
  <img src="images/RAGGPT UI.png" alt="RAG-GPT UI">
</div>

## Project Schema
<div align="center">
  <img src="images/RAGGPT_schema.png" alt="Schema">
</div>


## Document Storage
Documents are stored in two separate folders within the `data` directory:
- `data/docs_2`: For files that you want to **upload**.
- `data/docs`: For files that should be **processed in advance**.

## Server Setup
The `serve.py` module leverages these folders to create an **HTTPS server** that hosts the PDF files, making them accessible for user viewing.

## Database Creation
Vector databases (vectorDBs) are generated within the `data` folder, facilitating the project's functionality.

## Important Considerations
- The current file management system is intended for **demonstration purposes only**.
- It is **strongly recommended** to design a more robust and secure document handling process for any production deployment.
- Ensure that you place your files in the correct directories (`data/docs_2` and `data/docs`) for the project to function as intended.

```
Chat with your documents.
