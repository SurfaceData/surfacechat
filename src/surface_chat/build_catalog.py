# -*- coding: utf-8 -*-

import argparse
import chromadb

from chromadb.utils import embedding_functions
from gutenbergpy.gutenbergcache import GutenbergCache
from gutenbergpy.textget import get_text_by_id, strip_headers
from pydantic import BaseModel
from tqdm import tqdm
from typing import List

from surface_chat.gutenberg import BookRecord, Cache


def load_book(book: BookRecord, collection):
    BATCH_SIZE = 100
    for aid, author in enumerate(book.authors):
        documents = book.segments
        ids = [f"{book.id}:{aid}:{sid}" for sid in range(len(book.segments))]
        metadatas = [
            {
                "book_id": book.id,
                "author": author,
                "segment_id": sid,
            }
            for sid in range(len(book.segments))
        ]
        for b in range(0, len(ids), BATCH_SIZE):
            collection.add(
                ids=ids[b : b + BATCH_SIZE],
                metadatas=metadatas[b : b + BATCH_SIZE],
                documents=documents[b : b + BATCH_SIZE],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-url", type=str, default="")
    parser.add_argument("--index-name", type=str, default="sc_writing_style")
    parser.add_argument(
        "--embedding-model", type=str, default="multi-qa-MiniLM-L6-cos-v1"
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-segments", type=int, default=1000)
    parser.add_argument("--delete-old", action="store_true")
    args = parser.parse_args()

    client = chromadb.HttpClient(host=args.vector_url, port=443, ssl=True)
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

    author_counts = {}
    cache = Cache()
    books = cache.get_books(args.limit, args.offset)
    for book in tqdm(books):
        try:
            book.authors = cache.get_authors(book.id)
            book.titles = cache.get_title(book.id)
            text = strip_headers(get_text_by_id(book.id))
            book.segments = list(filter(str.strip, str(text[2000:-1000]).split("\\n")))[
                : args.max_segments
            ]
            load_book(book, collection)
            for author in book.authors:
                if author in author_counts:
                    author_counts[author] += 1
                else:
                    author_counts[author] = 1
        except Exception as e:
            pass
    sorted_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)
    print(sorted_authors[:20])
