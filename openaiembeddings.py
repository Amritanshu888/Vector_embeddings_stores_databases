#!/usr/bin/env python
# coding: utf-8

# ### Open AI Embeddings

# In[2]:


import os
from dotenv import load_dotenv
load_dotenv()


# In[4]:


os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")


# In[5]:


from langchain_openai import OpenAIEmbeddings
embeddings=OpenAIEmbeddings(model="text-embedding-3-small")


# In[6]:


embeddings


# In[7]:


## Single text embeddings
single_text="Langchain and Rag are amazing frameworks and projects to work on"
single_embeddings=embeddings.embed_query(single_text)
print(len(single_embeddings))
print(single_embeddings)


# In[8]:


print("📝 Single Text Embedding:")
print(f"Input: {single_text}")
print(f"Output: Vector of {len(single_embeddings)} dimensions")
print(f"Sample values: {single_embeddings[:5]}")


# In[9]:


# Example 2: Multiple texts at once
multiple_texts = [
    "Python is a programming language",
    "LangChain is a framework for LLM applications",
    "Embeddings convert text to numbers",
    "Vectors can be compared for similarity"
]


# In[10]:


multiple_embeddings = embeddings.embed_documents(multiple_texts)


# In[11]:


multiple_embeddings


# In[12]:


print("\n📚 Multiple Text Embeddings:")
print(f"Number of texts: {len(multiple_texts)}")
print(f"Number of embeddings: {len(multiple_embeddings)}")
print(f"Each embedding size: {len(multiple_embeddings[0])}")


# In[13]:


multiple_embeddings[0]


# In[14]:


from langchain_openai import OpenAIEmbeddings

# Different OpenAI embedding models
models_comparison = {
    "text-embedding-3-small": {
        "dimensions": 1536,
        "description": "Good balance of performance and cost",
        "cost_per_1m_tokens": 0.02,
        "use_case": "General purpose, cost-effective"
    },
    "text-embedding-3-large": {
        "dimensions": 3072,
        "description": "Highest quality embeddings",
        "cost_per_1m_tokens": 0.13,
        "use_case": "When accuracy is critical"
    },
    "text-embedding-ada-002": {
        "dimensions": 1536,
        "description": "Previous generation model",
        "cost_per_1m_tokens": 0.10,
        "use_case": "Legacy applications"
    }
}

# Display comparison
print("📊 OpenAI Embedding Models Comparison:\n")
for model_name, details in models_comparison.items():
    print(f"Model: {model_name}")
    print(f"  📏 Dimensions: {details['dimensions']}")
    print(f"  💰 Cost: ${details['cost_per_1m_tokens']}/1M tokens")
    print(f"  📝 Description: {details['description']}")
    print(f"  🎯 Use case: {details['use_case']}\n")


# ### Cosine Similarity With OpenAI Embeddings

# In[15]:


# Example 1: Finding similar sentences
sentences = [
    "The cat sat on the mat",
    "A feline rested on the rug",
    "The dog played in the yard",
    "I love programming in Python",
    "Python is my favorite programming language"
]


# In[16]:


import numpy as np
def cosine_similarity(vec1, vec2):
    """
    Cosine similarity measures the angle between two vectors.
    - Result close to 1: Very similar
    - Result close to 0: Not related
    - Result close to -1: Opposite meanings
    """

    dot_product=np.dot(vec1,vec2)
    norm_a=np.linalg.norm(vec1)
    norm_b=np.linalg.norm(vec2)
    return dot_product/(norm_a * norm_b)


# In[17]:


from langchain_openai import OpenAIEmbeddings
embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
embeddings


# In[18]:


sentence_embeddings=embeddings.embed_documents(sentences)
sentence_embeddings


# In[19]:


## Calculate the simialrity betwween all pairs

for i in range(len(sentences)):
    for j in range(i+1,len(sentences)):
        similarity=cosine_similarity(sentence_embeddings[i],sentence_embeddings[j])

        print(f"'{sentences[i]}' vs '{sentences[j]}'")
        print(f"Similarity: {similarity:.3f}\n")


# In[20]:


### Example- Semantic Search- Retireve the similar sentence
# Test semantic search
documents = [
    "LangChain is a framework for developing applications powered by language models",
    "Python is a high-level programming language",
    "Machine learning is a subset of artificial intelligence",
    "Embeddings convert text into numerical vectors",
    "The weather today is sunny and warm"
]
query="What is Langchain?"


# In[21]:


def semantic_search(query,documents,embeddings_models,top_k=3):
    """Simple semantic search implementation"""

    ## embed query and doument

    query_embedding=embeddings_models.embed_query(query)
    doc_embeddings = embeddings_models.embed_documents(documents)

    ## Calculate the similarity score

    similarties=[]

    for i,doc_emb in enumerate(doc_embeddings):
        similarity=cosine_similarity(query_embedding,doc_emb)
        similarties.append((similarity,documents[i]))

    ## Sort by similarity
    similarties.sort(reverse=True)
    return similarties[:top_k]



# In[22]:


results=semantic_search(query,documents,embeddings)
results


# In[23]:


print(f"\n🔎 Semantic Search Results for: '{query}'")
for score, doc in results:
    print(f"Score: {score:.3f} | {doc}")


# In[24]:


query="What is Embeddings?"
results=semantic_search(query,documents,embeddings)
results


# In[ ]:




