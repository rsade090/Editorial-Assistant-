import json
from pathlib import Path
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def chunker():
    
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    guidelines_path = Path("data/guidelines.json")
    articles_path   = Path("data/news-dataset-v2.json")
    output_dir      = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file     = output_dir / "chunks.jsonl"

    with guidelines_path.open("r", encoding="utf-8") as f:
        guidelines = json.load(f)
    with articles_path.open("r", encoding="utf-8") as f:
        articles = json.load(f)

    with output_file.open("w", encoding="utf-8") as out:
        
        for section in guidelines:
            text = section.get("content", "").strip()
            if not text:
                continue
            chunks = splitter.split_text(text)
            for idx, chunk in enumerate(chunks):
                record = {
                    "source": "guideline",
                    "content_section": section.get("content_section"),
                    "content_subsection": section.get("content_subsection"),
                    "url": section.get("url"),
                    "chunk_index": idx,
                    "chunk": chunk
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Process article descriptions
        for article in articles:
            text = article.get("body", "").strip()
            if not text:
                continue
            chunks = splitter.split_text(text)
            for idx, chunk in enumerate(chunks):
                record = {
                    "source": "article",
                    "content_id": article.get("content_id"),
                    "content_headline": article.get("content_headline"),
                    "content_publish_time": article.get("content_publish_time"),
                    "chunk_index": idx,
                    "chunk": chunk
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Chunks saved to {output_file} using LangChain TokenTextSplitter")

def build_faiss_index():

    chunks_path = Path("data/chunks.jsonl")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found at {chunks_path}")


    docs = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item.pop("chunk")
            metadata = item  
            docs.append(Document(page_content=text, metadata=metadata))

   
    embed_model = OpenAIEmbeddings()
    faiss_index = FAISS.from_documents(docs, embed_model)
    index_dir = Path("data")
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss_index.save_local(str(index_dir))
    print(f"FAISS index built and saved to {index_dir}")


if __name__ == "__main__":
    chunker()
    build_faiss_index()
