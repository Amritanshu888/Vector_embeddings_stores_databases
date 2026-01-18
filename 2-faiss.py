#!/usr/bin/env python
# coding: utf-8

# ### Building a RAG System with LangChain and FAISS 
# Introduction to RAG (Retrieval-Augmented Generation)
# RAG combines the power of retrieval systems with generative AI models. Instead of relying solely on the model's training data, RAG:
# 
# 1. Retrieves relevant documents from a knowledge base
# 2. Uses these documents as context for the LLM
# 3. Generates responses based on both the retrieved context and the model's knowledge

# ### FAISS 
# https://github.com/facebookresearch/faiss
# 
# FAISS is a library for efficient similarity search and clustering of dense vectors.
# 
# Key advantages:
# 1. Extremely fast similarity search
# 2. Memory efficient
# 3. Supports GPU acceleration
# 4. Can handle millions of vectors
# 
# How it works:
# - Indexes vectors for fast nearest neighbor search
# - Returns most similar vectors based on distance metrics
# 

# In[35]:


## load libraries
import os
from dotenv import load_dotenv
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# LangChain core imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough, 
 
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# LangChain specific imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()


# ### Data Ingestion And Processing
# 

# In[36]:


sample_documents = [
    Document(
        page_content="""
        Artificial Intelligence (AI) is the simulation of human intelligence in machines.
        These systems are designed to think like humans and mimic their actions.
        AI can be categorized into narrow AI and general AI.
        """,
        metadata={"source": "AI Introduction", "page": 1, "topic": "AI"}
    ),
    Document(
        page_content="""
        Machine Learning is a subset of AI that enables systems to learn from data.
        Instead of being explicitly programmed, ML algorithms find patterns in data.
        Common types include supervised, unsupervised, and reinforcement learning.
        """,
        metadata={"source": "ML Basics", "page": 1, "topic": "ML"}
    ),
    Document(
        page_content="""
        Deep Learning is a subset of machine learning based on artificial neural networks.
        It uses multiple layers to progressively extract higher-level features from raw input.
        Deep learning has revolutionized computer vision, NLP, and speech recognition.
        """,
        metadata={"source": "Deep Learning", "page": 1, "topic": "DL"}
    ),
    Document(
        page_content="""
        Natural Language Processing (NLP) is a branch of AI that helps computers understand human language.
        It combines computational linguistics with machine learning and deep learning models.
        Applications include chatbots, translation, sentiment analysis, and text summarization.
        """,
        metadata={"source": "NLP Overview", "page": 1, "topic": "NLP"}
    )
]

print(sample_documents)


# In[39]:


## text splitting
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=[" "]
)

## split the documents into chunks
chunks = text_splitter.split_documents(sample_documents)
print(chunks[0])
print(chunks[1])


# In[40]:


print(f"Created {len(chunks)} chunks from {len(sample_documents)} documents")
print("\nExample chunk:")
print(f"Content: {chunks[0].page_content}")
print(f"Metadata: {chunks[0].metadata}")


# In[41]:


### load the embedding models
import os
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")


# In[42]:


# Initialize OpenAI embeddings with the latest model

embeddings=OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536
)

## Example: create a embedding for a single text
sample_text="What is machine learning"
sample_embedding=embeddings.embed_query(sample_text)
sample_embedding


# In[44]:


texts=["AI","MAchine learning","Deep Learning","Neural Network"]
batch_embeddings=embeddings.embed_documents(texts)
print(batch_embeddings[0])


# In[45]:


print(batch_embeddings[1])


# In[46]:


### Compare Embedding using cosine similarity

def compare_embeddings(text1:str,text2:str):
    """Compare semantic simialrity of 2 texts usign embeddings"""

    emb1=np.array(embeddings.embed_query(text1))
    emb2=np.array(embeddings.embed_query(text2))

    ## Calculate the simialrity score

    similarity=np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity


# In[47]:


# Test semantic similarity
print("\nSemantic Similarity Examples:")
print(f"'AI' vs 'Artificial Intelligence': {compare_embeddings('AI', 'Artificial Intelligence'):.3f}")


# In[48]:


print(f"'AI' vs 'Pizza': {compare_embeddings('AI', 'Pizza'):.3f}")


# In[49]:


print(f"'Machine Learning' vs 'ML': {compare_embeddings('Machine Learning', 'ML'):.3f}")


# ### Create FAISS Vector Store

# In[50]:


vectorstore=FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
print(f"Vector store created with {vectorstore.index.ntotal} vectors")


# In[51]:


vectorstore


# In[52]:


## Save vector tore for later use
vectorstore.save_local("faiss_index")
print("Vector store saved to 'faiss_index' directory")


# In[53]:


## load vector store
loaded_vectorstore=FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"Loaded vector store contains {loaded_vectorstore.index.ntotal} vectors")


# In[54]:


## Similarity Search 
query="What is deep learning"

results=vectorstore.similarity_search(query,k=3)
print(results)


# In[55]:


print(f"Query: {query}\n")
print("Top 3 similar chunks:")
for i, doc in enumerate(results):
    print(f"\n{i+1}. Source: {doc.metadata['source']}")
    print(f"   Content: {doc.page_content[:200]}...")


# In[56]:


### Similarity Search with score
results_with_scores=vectorstore.similarity_search_with_score(query,k=3)

print("\n\nSimilarity search with scores:")
for doc, score in results_with_scores:
    print(f"\nScore: {score:.3f}")
    print(f"Source: {doc.metadata['source']}")
    print(f"Content preview: {doc.page_content[:100]}...")


# In[57]:


chunks


# In[59]:


### Search with metadata filtering
filter_dict={"topic":"ML"}
filtered_results=vectorstore.similarity_search(
    query,
    k=3,
    filter=filter_dict
)
print(filtered_results)


# In[60]:


len(filtered_results)


# ### Build RAG Chain With LCEL 

# In[75]:


## LLM GROQ LLM
from langchain.chat_models import init_chat_model

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

llm=init_chat_model(model="groq:gemma2-9b-it")
llm


# In[76]:


llm.invoke("Hi")


# In[62]:


# 1. Simple RAG Chain with LCEL
simple_prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:
Context: {context}

Question: {question}

Answer:""")


# In[64]:


## Basic retriever
retriever=vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)


# In[65]:


retriever


# In[ ]:


from typing import List
# Format documents for the prompt
def format_docs(docs: List[Document]) -> str:
    """Format documents for insertion into prompt"""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        formatted.append(f"Document {i+1} (Source: {source}):\n{doc.page_content}")
    return "\n\n".join(formatted)


# In[78]:


simple_rag_chain=(
    {"context":retriever | format_docs,"question":RunnablePassthrough() }
    | simple_prompt
    | llm
    |StrOutputParser()

)


# In[79]:


simple_rag_chain


# In[80]:


### Conversational RAg Chain

conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the provided context to answer questions."),
    ("placeholder", "{chat_history}"),
    ("human", "Context: {context}\n\nQuestion: {input}"),
])


# In[81]:


def create_conversational_rag():
    """Create a conversational RAG chain with memory"""
    return (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"]))
        )
        | conversational_prompt
        | llm
        | StrOutputParser()
    )

conversational_rag = create_conversational_rag()


# In[82]:


conversational_rag


# In[83]:


### streaming RAG chain
streaming_rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | simple_prompt
    | llm
)

print("Modern RAG chains created successfully!")
print("Available chains:")
print("- simple_rag_chain: Basic Q&A")
print("- conversational_rag: Maintains conversation history")
print("- streaming_rag_chain: Supports token streaming")


# In[86]:


# Test function for different chain types
def test_rag_chains(question: str):
    """Test all RAG chain variants"""
    print(f"Question: {question}")
    print("=" * 80)
    
    # 1. Simple RAG
    print("\n1. Simple RAG Chain:")
    answer = simple_rag_chain.invoke(question)
    print(f"Answer: {answer}")

    print("\n2. Streaming RAG:")
    print("Answer: ", end="", flush=True)
    for chunk in streaming_rag_chain.stream(question):
        print(chunk.content, end="", flush=True)
    print()


# In[87]:


test_rag_chains("What is the difference between AI and machine learning")


# In[88]:


# Test with multiple questions
test_questions = [
    "What is the difference between AI and Machine Learning?",
    "Explain deep learning in simple terms",
    "How does NLP work?"
]

for question in test_questions:
    print("\n" + "=" * 80 + "\n")
    test_rag_chains(question)


# In[89]:


## Conversational example
print("\n3. Conversational RAG Example:")
chat_history = []

# First question
q1 = "What is machine learning?"
a1 = conversational_rag.invoke({
    "input": q1,
    "chat_history": chat_history
})

print(f"Q1: {q1}")
print(f"A1: {a1}")


# In[90]:


# Update history
chat_history.extend([
    HumanMessage(content=q1),
    AIMessage(content=a1)
])


# In[91]:


# Follow-up question
q2 = "How is it different from traditional programming?"
a2 = conversational_rag.invoke({
    "input": q2,
    "chat_history": chat_history
})
print(f"\nQ2: {q2}")
print(f"A2: {a2}")


# In[ ]:




