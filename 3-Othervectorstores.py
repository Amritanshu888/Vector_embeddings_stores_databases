#!/usr/bin/env python
# coding: utf-8

# ### InMemoryVectorStore
# In-memory vector store implementation.
# 
# Uses a dictionary, and computes cosine similarity for search using numpy.

# In[1]:


import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

from langchain.chat_models import init_chat_model

llm=init_chat_model("openai:gpt-4o-mini")
llm


# In[5]:


from langchain_openai import OpenAIEmbeddings

from langchain_core.vectorstores import InMemoryVectorStore

vector_store=InMemoryVectorStore(embedding=OpenAIEmbeddings())


# In[ ]:


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


# In[3]:


documents


# In[6]:


vector_store.add_documents(documents=documents)


# In[7]:


vector_store.similarity_search("hows the weather forecast")


# In[12]:


vector_store.similarity_search("hows the weather forecast",k=2)


# In[15]:


### vectorstore to retriever

retriever=vector_store.as_retriever(search_kwargs={"k":2})

retriever


# In[ ]:


## Invoke
retriever.invoke("hows the weather forecast")


# In[ ]:




