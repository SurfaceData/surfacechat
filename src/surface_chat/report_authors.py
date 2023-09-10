# -*- coding: utf-8 -*-

import argparse
import json
import re

from datasets import load_dataset


def get_name(authors: str):
    distinct_names = authors.split(";")
    name_parts = distinct_names[0].split(",")
    if len(name_parts) < 2:
        return distinct_names[0].strip()
    return " ".join([name_parts[1], name_parts[0]]).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", type=int, default=-1)
    args = parser.parse_args()

    dataset = load_dataset("sedthh/gutenberg_english", split=f"train[:{args.end}]")
    examples = dataset["TEXT"]
    metadatas = dataset["METADATA"]
    authors = {}
    for i, m in enumerate(metadatas):
        metadata = json.loads(m)
        author = get_name(metadata["authors"])
        clean_text = re.sub(r"\r\n\r", " ", examples[i])
        clean_text = re.sub(r"\n", " ", clean_text)
        clean_text = re.sub("\xa0", " ", clean_text)
        documents = [s.strip() for s in clean_text.split("\r")][:500]
        if author in authors:
            authors[author] += len(documents)
        else:
            authors[author] = len(documents)
    sorted_authors = sorted(authors.items(), key=lambda x: x[1], reverse=True)
    sorted_author_names = [name for name, count in sorted_authors]
    print(sorted_author_names[:50])
    for name in sorted_author_names[:50]:
        print(name)
        print(authors[name])
