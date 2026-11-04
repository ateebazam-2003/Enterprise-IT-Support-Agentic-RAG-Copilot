
from app.core.config import get_settings
from pathlib import Path
from app.services.ingestion import load_file,chunking 
from app.rag.vector_store import add_document


settings=get_settings()

folder=Path(settings.sample_kb_dir)
files=[p for p in folder.iterdir() if p.is_file()]
all_docs=[]


for path in files:
    
    all_docs.extend(load_file(path))
chunks=chunking(all_docs)
ids=add_document(chunks)

print(f"Indexed:{(len(files))} files\nLength:{len(chunks)}\nId:{len(ids)} PineCone Vectors")

