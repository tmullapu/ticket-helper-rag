# Ticket Helper

A small RAG (retrieval-augmented generation) learning project. Given a support ticket, it finds the most relevant internal docs (SOPs, past incidents, how-tos) and asks a local LLM to suggest a fix, citing the sources it used. If nothing relevant is found, it answers "I don't know" instead of guessing.

## How it works

1. **Load:** read the `.txt` files in `docs/`.
2. **Split:** break them into chunks of 600 characters with 60 characters of overlap.
3. **Embed:** turn each chunk into a vector with Voyage AI (`voyage-4-lite`).
4. **Store:** save the vectors, text and source file names in a local Chroma database (`chroma_db/`).
5. **Retrieve:** embed the ticket with the same model and fetch the 3 closest chunks.
6. **Answer:** if the best match is close enough, send those chunks and the ticket to Llama 3.2 3B running locally in Ollama. Otherwise print "I don't know".

`ingest.py` runs steps 1 to 4 and `ask.py` runs steps 5 and 6.

## Tools used

- [LangChain](https://python.langchain.com/) document loaders and text splitter
- [Voyage AI](https://www.voyageai.com/) embeddings
- [Chroma](https://www.trychroma.com/) vector database
- [Ollama](https://ollama.com/) with `llama3.2:3b`
- `python-dotenv` to load the API key

## Running it

You need Python 3, [Ollama](https://ollama.com/) and a Voyage AI API key.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then put your real key in .env
ollama pull llama3.2:3b

python ingest.py            # build the vector store (run again after changing docs/)
python ask.py               # answer the sample ticket
```

To try a different ticket, edit the `ticket` variable in `ask.py`.

## What I learned

- **Chunk size matters.** If chunks are too small, a fix gets split across pieces and loses context. If they're too large, unrelated content dilutes the embedding. 600 characters with a small overlap worked well for short SOP-style docs.
- **A distance threshold lets the app say "I don't know".** Chroma always returns the nearest chunks, even when none of them are relevant. Checking the best distance against a threshold (`THRESHOLD = 1.2`) stops the model from making up an answer from unrelated context. I tuned the value by trying tickets that should and shouldn't match.
- **Model size is a tradeoff.** A small local model (3B) is fast and free to run, but it follows instructions less reliably, for example when citing sources or knowing when to say "I don't know". Clear prompts and good retrieval matter more with a small model.
