import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from app.core.config import get_settings


settings=get_settings()

_embedding=None
_vector_Store=None


DIMENSIONS={
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

def get_dimension(model_name:str | None=None):
    name=(model_name or settings.embedding_model or "").strip()
    
    if not name:
        raise RuntimeError("Embedding Model Not Configured")
    
    if name in DIMENSIONS:
        return DIMENSIONS[name]
    else:
        raise ValueError(
    f"Unsupported embedding model '{model_name or settings.embedding_model}' for Pinecone. "
    "Add the matching dimension to EMBEDDING_DIMENSIONS.")



def get_embedding():
    global _embedding
    
    if _embedding is None:
        if  settings.embedding_model=="":
            raise ValueError("Embedding Model not Configured")
        else:
            _embedding=HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                encode_kwargs={'normalize_embeddings':True}
            )
    return _embedding


def ensure_index():
    if not settings.pinecone_api_key:
        raise ValueError("PineCone Api Key not Cinfigured!")
    
    desired_dimension=get_dimension()
    pc=Pinecone(api_key=settings.pinecone_api_key)
    
    names=[x["name"] for x in pc.list_indexes()]
    
    if settings.pinecone_index_name in names:
        index_info=pc.describe_index(settings.pinecone_index_name)
        current_dimension=getattr(index_info,"dimension",None)
        
        if current_dimension is None and isinstance(index_info,dict):
            current_dimension=index_info.get("dimension")
        
        if current_dimension is not None and current_dimension!=desired_dimension:
            pc.delete_index(settings.pinecone_index_name)
            
            while settings.pinecone_index_name in [x["name"] for x in pc.list_indexes()]:
                time.sleep(1)
    
    if settings.pinecone_index_name not in names:
        pc.create_index(
            settings.pinecone_index_name,
            dimension=desired_dimension,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws',region='us-east-1')
        )
        
        while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
            time.sleep(1)
    return pc.Index(settings.pinecone_index_name)
        
    
    

def get_vectore_store():
    global _vector_Store
    
    if _vector_Store is None:
        index=ensure_index()
        
        _vector_Store=PineconeVectorStore(
            index=index,
            embedding=get_embedding(),
            namespace=settings.pinecone_namespace
        )
    return _vector_Store


def get_retriever():
    return get_vectore_store().as_retriever(search_kwargs={'k':settings.top_k})

def add_document(chunks):
    store=get_vectore_store()
    return store.add_documents(chunks)

