This is a RAG agent in voice that gives retreived information based on provided document within 700ms. 
Tested with 750 page document with accurate answers.

Commands to run the project:

1. uv sync
2. uv run document_indexer.py --> to index the doc
3. uv run agent.py dev --> for dev mode with console logs

Note:
- .env keys require a Qdrant, Groq, Deepgram and Livekit account.
- To add a document, create a "doc" folder in the root and place your PDF or .docx in the folder.
