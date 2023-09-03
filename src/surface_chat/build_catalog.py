# -*- coding: utf-8 -*-

from gutenbergpy.gutenbergcache import GutenbergCache
from gutenbergpy.textget import get_text_by_id, strip_headers
from pydantic import BaseModel
from tqdm import tqdm
from typing import List


class BookRecord(BaseModel):
    id: int
    authors: List[str] = []
    titles: List[str] = []
    segments: List[str] = []


cache = GutenbergCache.get_cache()


def get_authors(book_id: int):
    cursor = cache.native_query(
        f"""
        SELECT
          a.name
        FROM
          book_authors AS ba
        INNER JOIN 
          authors AS a
        ON a.id = ba.authorid
        WHERE ba.bookid = {book_id}
    """
    )
    return [a[0] for a in cursor]


def get_title(book_id: int):
    cursor = cache.native_query(
        f"""
        SELECT
          name
        FROM
          titles
        WHERE bookid = {book_id}
    """
    )
    return [t[0] for t in cursor]


book_cursor = cache.native_query(
    """
    SELECT
      *
    FROM
      books
"""
)


author_counts = {}
books = [BookRecord(id=b[0]) for b in book_cursor]
for book in tqdm(books):
    try:
        book.authors = get_authors(book.id)
        book.titles = get_title(book.id)
        # text = strip_headers(get_text_by_id(book.id))
        # book.segments = list(filter(str.strip, str(text[2000:-1000]).split("\\n")))

        for author in book.authors:
            if not author in author_counts:
                author_counts[author] = 0
            author_counts[author] += 1
    except Exception as e:
        print(e)

author_count_list = [(k, v) for k, v in author_counts.items()]
author_count_list = sorted(author_count_list, key=lambda x: x[1])
print(author_count_list)
