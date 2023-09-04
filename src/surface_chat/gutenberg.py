from gutenbergpy.gutenbergcache import GutenbergCache
from pydantic import BaseModel
from typing import List


class BookRecord(BaseModel):
    id: int
    authors: List[str] = []
    titles: List[str] = []
    segments: List[str] = []


class Cache:
    def __init__(self):
        self.cache = GutenbergCache.get_cache()

    def get_books(self, limit: int, offset: int):
        book_cursor = self.cache.native_query(
            f"""
            SELECT
              *
            FROM
              books
            LIMIT {limit}
            OFFSET {offset}
        """
        )
        return [BookRecord(id=b[0]) for b in book_cursor]

    def get_authors(self, book_id: int):
        cursor = self.cache.native_query(
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

    def get_title(self, book_id: int):
        cursor = self.cache.native_query(
            f"""
            SELECT
              name
            FROM
              titles
            WHERE bookid = {book_id}
        """
        )
        return [t[0] for t in cursor]
