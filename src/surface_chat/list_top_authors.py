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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    author_counts = {}
    cache = Cache()
    books = cache.get_books(args.limit, args.offset)
    for book in tqdm(books):
        try:
            book.authors = cache.get_authors(book.id)
            for author in book.authors:
                if author in author_counts:
                    author_counts[author] += 1
                else:
                    author_counts[author] = 1
        except Exception as e:
            pass
    sorted_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)
    print(sorted_authors[: args.top_k])
