# Retrieval-Augmented Generation (RAG) Bootstrap Application UI

This is a [LlamaIndex](https://www.llamaindex.ai/) project bootstrapped with [`create-llama`](https://github.com/run-llama/LlamaIndexTS/tree/main/packages/create-llama) to act as a full stack UI to accompany **Retrieval-Augmented Generation (RAG) Bootstrap Application**, which can be found in its own repository at https://github.com/tyrell/llm-ollama-llamaindex-bootstrap 


![UI Screenshot](https://github.com/tyrell/llm-ollama-llamaindex-bootstrap-ui/blob/main/screenshots/ui-screenshot.png?raw=true)


The backend code of this application has been modified as below;

1. Loading the Vector Store Index created previously in the **Retrieval-Augmented Generation (RAG) Bootstrap Application** in response to user queries submitted through the frontend UI.
   -   Refer backend/app/utils/index.py and the code comments to understand the modifications.
2. Querying the index with streaming enabled 
   -   Refer backend/app/api/routers/chat.py and the code comments to understand the modifications.

## Running the full stack application

First, startup the backend as described in the [backend README](./backend/README.md).

Second, run the development server of the frontend as described in the [frontend README](./frontend/README.md).

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

# License

Apache 2.0

~ Tyrell Perera 
