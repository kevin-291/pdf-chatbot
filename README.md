# PDF Chatbot

This project implements a Streamlit-based chatbot that can answer questions based on the content of uploaded PDF documents. It uses LangChain for document processing and retrieval, and Ollama for the language model.

## Features

- PDF upload and processing
- Conversational interface for asking questions about the PDF content
- Uses RAG (Retrieval-Augmented Generation) for accurate answers
- Supports GPU acceleration (if available)

## Requirements

- Python 3.7+
- CUDA-compatible GPU (optional, for faster processing)

See `requirements.txt` for a full list of Python dependencies.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/kevin-291/pdf-chatbot.git
   cd pdf-chatbot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have Ollama installed.

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3. Use the sidebar to upload a PDF file and process it.

4. Once processed, you can start asking questions about the PDF content in the chat interface.

## How it Works

1. **PDF Processing**: The app uses PyPDFLoader to read the PDF and split it into chunks using `RecursiveCharacterTextSplitter`.

2. **Embedding**: Document chunks are embedded using the `sentence-transformers/all-MiniLM-L6-v2` model.

3. **Vector Store**: FAISS is used to create a vector store from the embedded documents.

4. **Question Answering**: When a user asks a question, the app uses a `ConversationalRetrievalChain` to:
   - Retrieve relevant document chunks
   - Generate a response using the Ollama language model
   - Maintain conversation history

5. **User Interface**: Streamlit is used to create an interactive web interface for PDF upload and chatting.

## Customization

- To use a different language model, change the `MODEL` variable in `app.py`.
- Adjust the `chunk_size` and `chunk_overlap` in the `RecursiveCharacterTextSplitter` for different document splitting behavior.
- Modify the prompt template in `app.py` to change the AI assistant's behavior.

## Limitations

- The accuracy of answers depends on the quality of the uploaded PDF and the capabilities of the chosen language model.
- Large PDFs may take some time to process.
- The app currently supports only one PDF at a time.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).