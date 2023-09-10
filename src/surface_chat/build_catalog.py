# -*- coding: utf-8 -*-

import argparse
import chromadb
import json
import re

from chromadb.utils import embedding_functions
from datasets import load_dataset


def get_name(authors: str):
    distinct_names = authors.split(";")
    name_parts = distinct_names[0].split(",")
    if len(name_parts) < 2:
        return distinct_names[0].strip()
    return " ".join([name_parts[1], name_parts[0]]).strip()


DROP_WORDS = [
    "CHAPTER",
    "[Illustration]",
]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-url", type=str, default="")
    parser.add_argument("--vector-port", type=int, default=443)
    parser.add_argument("--vector-ssl", action="store_true")
    parser.add_argument("--index-name", type=str, default="sc_writing_style")
    parser.add_argument("--delete-old", action="store_true")
    parser.add_argument(
        "--embedding-model", type=str, default="multi-qa-MiniLM-L6-cos-v1"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    client = chromadb.HttpClient(
        host=args.vector_url, port=args.vector_port, ssl=args.vector_ssl
    )
    if args.delete_old:
        collections = client.list_collections()
        for collection in collections:
            if collection.name == args.index_name:
                print("Deleting old collection")
                client.delete_collection(args.index_name)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=args.embedding_model, device=args.device
    )
    collection = client.get_or_create_collection(
        name=args.index_name, embedding_function=embedding_fn
    )

    batch_size = args.batch_size

    def accept_snippet(s: str):
        if len(s) < 50:
            return False
        for word in DROP_WORDS:
            if word in s:
                return False
        return True

    def load_book(example):
        metadata = json.loads(example["METADATA"])
        text_id = metadata["text_id"]
        author = get_name(metadata["authors"])
        title = metadata["title"]
        if author == "":
            return example
        clean_text = re.sub(r"\r\n\r", " ", example["TEXT"])
        clean_text = re.sub(r"\n", " ", clean_text)
        clean_text = re.sub("\xa0", " ", clean_text)
        documents = [s.strip() for s in clean_text.split("\r")]
        documents = list(filter(accept_snippet, documents))[: args.limit]
        ids = [f"{text_id}:{sid}" for sid in range(len(documents))]
        metadatas = [
            {
                "text_id": text_id,
                "sid": sid,
                "author": author,
                "title": title,
            }
            for sid in range(len(documents))
        ]
        print(f"Adding {len(documents)} entries for {author}")
        for b in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[b : b + batch_size],
                metadatas=metadatas[b : b + batch_size],
                documents=documents[b : b + batch_size],
            )
        return example

    dataset = load_dataset(
        "sedthh/gutenberg_english", split=f"train[{args.start}:{args.end}]"
    )
    dataset.map(load_book)
