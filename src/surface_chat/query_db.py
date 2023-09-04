# -*- coding: utf-8 -*-

import argparse

from tqdm import tqdm

from surface_chat.gutenberg import BookRecord, Cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bookid", type=int, default=0)
    args = parser.parse_args()

    cache = Cache()
    print(cache.get_authors(args.bookid))
