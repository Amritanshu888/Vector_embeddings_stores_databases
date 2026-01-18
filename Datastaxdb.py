#!/usr/bin/env python
# coding: utf-8

# ### ASTRADB VectorStore
# Go from app idea to production with the AI Platform with Astra DB, the ultra-low latency database made for AI and Langflow, the low-code RAG IDE
# https://www.datastax.com/

# In[20]:


### ASTRADB VectorStore


# In[1]:


get_ipython().system('pip install      "langchain>=0.3.23,<0.4"      "langchain-core>=0.3.52,<0.4"      "langchain-astradb>=0.6,<0.7"')


# In[1]:


get_ipython().system('pip install langchain_openai')


# In[22]:


### Config
ASTRA_DB_API_ENDPOINT="https://1beaf365-fca9-432e-8d11-e6bd566158-east-2.apps.astra.datastax.com"
ASTRA_DB_APPLICATION_TOKEN="AstraCS:UQibHoJlupgvUnACFizHwATs1745618f6b2a9046b25a72378dab3f83869a940f4015681811ddb67cc438f1"


# In[21]:


from langchain_openai import OpenAIEmbeddings
embeddings=OpenAIEmbeddings(model="text-embedding-3-small",dimensions=1024,api_key="sk-proj-rGVOWIqWsV5LRRgRf0dTD601UhLQ5mdV9eKVlyvT32EUBmbSxCsBQ9Q9sIxVm40edITMtT3BlbkFJCZIyoHYPvNm2FMGdk7zHMyi20wNH-IlbhDL_iuveWaOdyiCeoFDAiK-k9iJ4YQ_PrOIsaSGogA")


# In[9]:


embeddings


# In[10]:


from langchain_astradb import AstraDBVectorStore
vector_store=AstraDBVectorStore(
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    collection_name="astra_vector_langchain",
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=None,

)
vector_store


# In[11]:


from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
documents


# In[13]:


vector_store.add_documents(documents=documents)


# In[14]:


### Search from Vector Store DB

vector_store.similarity_search("What is the weather")


# In[15]:


results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=3,
    filter={"source": "tweet"},
)
for res in results:
    print(f'* "{res.page_content}", metadata={res.metadata}')


# In[16]:


results = vector_store.similarity_search_with_score(
    "LangChain provides abstractions to make working with LLMs easy",
    k=3,
    filter={"source": "tweet"},
)
for res, score in results:
    print(f'* [SIM={score:.2f}] "{res.page_content}", metadata={res.metadata}')


# In[17]:


### Retriever
retriever=vector_store.as_retriever(
  search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.5},
)
retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})


# In[18]:


### Retriever
retriever=vector_store.as_retriever(
  search_type="mmr",
    search_kwargs={"k": 1},
)
retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})


# In[ ]:




