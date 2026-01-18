#!/usr/bin/env python
# coding: utf-8

# ### Building a RAG System with LangChain and ChromaDB
# #### Introduction
# Retrieval-Augmented Generation (RAG) is a powerful technique that combines the capabilities of large language models with external knowledge retrieval. This notebook will walk you through building a complete RAG system using:
# 
# - LangChain: A framework for developing applications powered by language models
# - ChromaDB: An open-source vector database for storing and retrieving embeddings
# - OpenAI: For embeddings and language model (you can substitute with other providers)

# In[2]:


import os
from dotenv import load_dotenv
load_dotenv()


# In[3]:


## langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

## vectorstores
from langchain_community.vectorstores import Chroma

## utility imports
import numpy as np
from typing import List


# In[4]:


# RAG Architecture Overview
print("""
RAG (Retrieval-Augmented Generation) Architecture:

1. Document Loading: Load documents from various sources
2. Document Splitting: Break documents into smaller chunks
3. Embedding Generation: Convert chunks into vector representations
4. Vector Storage: Store embeddings in ChromaDB
5. Query Processing: Convert user query to embedding
6. Similarity Search: Find relevant chunks from vector store
7. Context Augmentation: Combine retrieved chunks with query
8. Response Generation: LLM generates answer using context

Benefits of RAG:
- Reduces hallucinations
- Provides up-to-date information
- Allows citing sources
- Works with domain-specific knowledge
""")


# ### 1. Sample Data

# In[5]:


## create sample documents
sample_docs = [
    """
    Machine Learning Fundamentals
    
    Machine learning is a subset of artificial intelligence that enables systems to learn 
    and improve from experience without being explicitly programmed. There are three main 
    types of machine learning: supervised learning, unsupervised learning, and reinforcement 
    learning. Supervised learning uses labeled data to train models, while unsupervised 
    learning finds patterns in unlabeled data. Reinforcement learning learns through 
    interaction with an environment using rewards and penalties.
    """,
    
    """
    Deep Learning and Neural Networks
    
    Deep learning is a subset of machine learning based on artificial neural networks. 
    These networks are inspired by the human brain and consist of layers of interconnected 
    nodes. Deep learning has revolutionized fields like computer vision, natural language 
    processing, and speech recognition. Convolutional Neural Networks (CNNs) are particularly 
    effective for image processing, while Recurrent Neural Networks (RNNs) and Transformers 
    excel at sequential data processing.
    """,
    
    """
    Natural Language Processing (NLP)
    
    NLP is a field of AI that focuses on the interaction between computers and human language. 
    Key tasks in NLP include text classification, named entity recognition, sentiment analysis, 
    machine translation, and question answering. Modern NLP heavily relies on transformer 
    architectures like BERT, GPT, and T5. These models use attention mechanisms to understand 
    context and relationships between words in text.
    """
]

sample_docs


# In[6]:


## save sample documents to files
import tempfile
temp_dir=tempfile.mkdtemp()

for i,doc in enumerate(sample_docs):
    with open(f"{temp_dir}/doc_{i}.txt","w") as f:
        f.write(doc)

print(f"Sample document create in : {temp_dir}")


# In[7]:


## save sample documents to files
import tempfile
temp_dir=tempfile.mkdtemp()

for i,doc in enumerate(sample_docs):
    with open(f"doc_{i}.txt","w") as f:
        f.write(doc)



# In[7]:


temp_dir


# ### 2. Document Loading

# In[8]:


from langchain_community.document_loaders import DirectoryLoader,TextLoader

# Load documents from directory
loader = DirectoryLoader(
    "data", 
    glob="*.txt", 
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)
documents = loader.load()

print(f"Loaded {len(documents)} documents")
print(f"\nFirst document preview:")
print(documents[0].page_content[:200] + "...")


# In[9]:


documents


# ### Document Splitting

# In[10]:


# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Maximum size of each chunk
    chunk_overlap=50,  # Overlap between chunks to maintain context
    length_function=len,
    separators=[" "]  # Hierarchy of separators
)
chunks=text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks from {len(documents)} documents")
print(f"\nChunk example:")
print(f"Content: {chunks[0].page_content[:150]}...")
print(f"Metadata: {chunks[0].metadata}")


# In[11]:


chunks


# ### Embedding Models

# In[12]:


os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")


# In[13]:


sample_text="MAchine LEarning is fascinating"
embeddings=OpenAIEmbeddings()
embeddings


# In[14]:


vector=embeddings.embed_query(sample_text)
vector


# ### Intilialize the ChromaDB Vector Store And Stores the chunks in Vector Representation

# In[15]:


chunks


# In[16]:


## Create a Chromdb vector store
persist_directory="./chroma_db"

## Initialize Chromadb with Open AI embeddings
vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory=persist_directory,
    collection_name="rag_collection"

)

print(f"Vector store created with {vectorstore._collection.count()} vectors")
print(f"Persisted to: {persist_directory}")


# ### Test Similarity Search

# In[17]:


query="What are the types of machine learning?"

similar_docs=vectorstore.similarity_search(query,k=3)
similar_docs


# In[18]:


query="what is NLP?"

similar_docs=vectorstore.similarity_search(query,k=3)
similar_docs


# In[19]:


query="what is Deep Learning?"

similar_docs=vectorstore.similarity_search(query,k=3)
similar_docs


# In[20]:


print(f"Query: {query}")
print(f"\nTop {len(similar_docs)} similar chunks:")
for i, doc in enumerate(similar_docs):
    print(f"\n--- Chunk {i+1} ---")
    print(doc.page_content[:200] + "...")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")


# ### Advanced Similarity Search With Scores

# In[21]:


results_scores=vectorstore.similarity_search_with_score(query,k=3)
results_scores


# #### Understanding Similarity Scores
# The similarity score represents how closely related a document chunk is to your query. The scoring depends on the distance metric used:
# 
# ChromaDB default: Uses L2 distance (Euclidean distance)
# 
# - Lower scores = MORE similar (closer in vector space)
# - Score of 0 = identical vectors
# - Typical range: 0 to 2 (but can be higher)
# 
# 
# Cosine similarity (if configured):
# 
# - Higher scores = MORE similar
# - Range: -1 to 1 (1 being identical)

# #### Initialize LLM, RAG Chain, Prompt Template,Query the RAG system

# In[22]:


from langchain_openai import ChatOpenAI

llm=ChatOpenAI(
    model_name="gpt-3.5-turbo"
)


# In[23]:


test_response=llm.invoke("What is Large Language Models")
test_response


# In[24]:


from langchain.chat_models.base import init_chat_model

llm=init_chat_model("openai:gpt-3.5-turbo")
#llm=init_chat_model("groq:")
llm


# In[25]:


llm.invoke("What is AI")


# ### Modern RAG Chain

# In[26]:


from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


# In[27]:


## Convert vector store to retriever
retriever=vectorstore.as_retriever(
    search_kwarg={"k":3} ## Retrieve top 3 relevant chunks
)
retriever


# In[28]:


## Create a prompt template
from langchain_core.prompts import ChatPromptTemplate
system_prompt="""You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


# In[29]:


prompt


# ##### What is create_stuff_documents_chain?
# create_stuff_documents_chain creates a chain that "stuffs" (inserts) all retrieved documents into a single prompt and sends it to the LLM. It's called "stuff" because it literally stuffs all the documents into the context window at once.

# In[30]:


### Create a document chain
from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain=create_stuff_documents_chain(llm,prompt)
document_chain


# This chain:
# 
# - Takes retrieved documents
# - "Stuffs" them into the prompt's {context} placeholder
# - Sends the complete prompt to the LLM
# - Returns the LLM's response

# #### What is create_retrieval_chain?
# create_retrieval_chain is a function that combines a retriever (which fetches relevant documents) with a document chain (which processes those documents with an LLM) to create a complete RAG pipeline.

# In[31]:


### Create The Final RAG Chain
from langchain.chains import create_retrieval_chain
rag_chain=create_retrieval_chain(retriever,document_chain)
rag_chain


# In[ ]:





# In[32]:


response=rag_chain.invoke({"input":"What is Deep LEarning"})


# In[33]:


response


# In[34]:


response['answer']


# In[35]:


# Function to query the modern RAG system
def query_rag_modern(question):
    print(f"Question: {question}")
    print("-" * 50)
    
    # Using create_retrieval_chain approach
    result = rag_chain.invoke({"input": question})
    
    print(f"Answer: {result['answer']}")
    print("\nRetrieved Context:")
    for i, doc in enumerate(result['context']):
        print(f"\n--- Source {i+1} ---")
        print(doc.page_content[:200] + "...")
    
    return result

# Test queries
test_questions = [
    "What are the three types of machine learning?",
    "What is deep learning and how does it relate to neural networks?",
    "What are CNNs best used for?"
]

for question in test_questions:
    result = query_rag_modern(question)
    print("\n" + "="*80 + "\n")


# ### Create RAG Chain Alternative - Using LCEL (LangChain Expression Language)

# In[38]:


# Even more flexible approach using LCEL
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


# In[39]:


# Create a custom prompt
custom_prompt = ChatPromptTemplate.from_template("""Use the following context to answer the question. 
If you don't know the answer based on the context, say you don't know.
Provide specific details from the context to support your answer.

Context:
{context}

Question: {question}

Answer:""")
custom_prompt


# In[40]:


retriever


# In[41]:


## Format the output documents for the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# In[62]:


## Build the chain ussing LCEL

rag_chain_lcel=(
    { 
        "context":retriever | format_docs,
        "question": RunnablePassthrough()
     }
    | custom_prompt
    | llm
    | StrOutputParser()
)

rag_chain_lcel


# In[63]:


response=rag_chain_lcel.invoke("What is Deep Learning")
response


# In[64]:


retriever.get_relevant_documents("What is Deep Learning")


# In[65]:


# Query using the LCEL approach - Fixed version
def query_rag_lcel(question):
    print(f"Question: {question}")
    print("-" * 50)
    
    # Method 1: Pass string directly (when using RunnablePassthrough)
    answer = rag_chain_lcel.invoke(question)
    print(f"Answer: {answer}")
    
    # Get source documents separately if needed
    docs = retriever.get_relevant_documents(question)
    print("\nSource Documents:")
    for i, doc in enumerate(docs):
        print(f"\n--- Source {i+1} ---")
        print(doc.page_content[:200] + "...")


# In[66]:


# Test LCEL chain
print("Testing LCEL Chain:")
query_rag_lcel("What are the key concepts in reinforcement learning?")


# In[67]:


query_rag_lcel("What is machine learning?")


# In[69]:


query_rag_lcel("What is depe learning?")


# ### Add New Documents To Existing Vector Store

# In[73]:


vectorstore


# In[74]:


# Add new documents to the existing vector store
new_document = """
Reinforcement Learning in Detail

Reinforcement learning (RL) is a type of machine learning where an agent learns to make 
decisions by interacting with an environment. The agent receives rewards or penalties 
based on its actions and learns to maximize cumulative reward over time. Key concepts 
in RL include: states, actions, rewards, policies, and value functions. Popular RL 
algorithms include Q-learning, Deep Q-Networks (DQN), Policy Gradient methods, and 
Actor-Critic methods. RL has been successfully applied to game playing (like AlphaGo), 
robotics, and autonomous systems.
"""


# In[75]:


new_document


# In[76]:


chunks


# In[77]:


new_doc=Document(
    page_content=new_document,
    metadata={"source": "manual_addition", "topic": "reinforcement_learning"}
)


# In[79]:


new_doc


# In[81]:


## split the documents
new_chunks=text_splitter.split_documents([new_doc])
new_chunks


# In[82]:


### Add new documents to vectorstore
vectorstore.add_documents(new_chunks)



# In[83]:


print(f"Added {len(new_chunks)} new chunks to the vector store")
print(f"Total vectors now: {vectorstore._collection.count()}")


# In[84]:


## query with the updated vector
new_question="What are the keys concepts in reinforcement learning"
result=query_rag_lcel(new_question)
result


# ### Advanced Rag Techniques- Conversational Memory
# Understanding Conversational Memory in RAG
# Conversational memory enables RAG systems to maintain context across multiple interactions. This is crucial for:
# 
# Follow-up questions that reference previous answers
# Pronoun resolution (e.g., "it", "they", "that")
# Context-dependent queries that build on prior discussion
# Natural dialogue flow where users don't repeat context
# 
# Key Challenge:
# Traditional RAG retrieves documents based only on the current query, missing important context from the conversation. For example:
# 
# User: "Tell me about Python"
# Bot: explains Python programming language
# User: "What are its main libraries?" ← "its" refers to Python, but retriever doesn't know this
# 
# Solution:
# The modern approach uses a two-step process:
# 
# Query Reformulation: Transform context-dependent questions into standalone queries
# Context-Aware Retrieval: Use the reformulated query to fetch relevant documents

# - create_history_aware_retriever: Makes the retriever understand conversation context
# - MessagesPlaceholder: Placeholder for chat history in prompts
# - HumanMessage/AIMessage: Structured message types for conversation history

# In[85]:


from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# In[86]:


## create a prompt that includes the chat history
contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# In[87]:


## create history aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
history_aware_retriever


# In[88]:


# Create a new document chain with history
qa_system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create conversational RAG chain
conversational_rag_chain = create_retrieval_chain(
    history_aware_retriever, 
    question_answer_chain
)
print("Conversational RAG chain created!")


# In[89]:


chat_history=[]
# First question
result1 = conversational_rag_chain.invoke({
    "chat_history": chat_history,
    "input": "What is machine learning?"
})
print(f"Q: What is machine learning?")
print(f"A: {result1['answer']}")


# In[91]:


chat_history.extend([
    HumanMessage(content="What is machine learning"),
    AIMessage(content=result1['answer'])
])


# In[92]:


chat_history


# In[93]:


## Follow up question
# Follow-up question
result2 = conversational_rag_chain.invoke({
    "chat_history": chat_history,
    "input": "What are its main types?"  # Refers to ML from previous question
})
result2


# In[94]:


result2['answer']


# ### Using GROQ LLM's
#  

# In[97]:


llm


# In[98]:


load_dotenv()


# In[99]:


os.getenv("GROQ_API_KEY")


# In[100]:


from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model


# In[101]:


os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


# In[102]:


llm=ChatGroq(model="gemma2-9b-it",api_key=os.getenv("GROQ_API_KEY"))
llm


# In[103]:


llm=init_chat_model(model="groq:gemma2-9b-it")
llm


# In[ ]:




