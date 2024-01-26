#!/usr/bin/env python3
"""Ingest bookmarks from specific folder."""

import configparser
import os
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from dotenv import load_dotenv

from langchain_community.document_loaders.html import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from constants import CHROMA_SETTINGS

from ingest import chunk_overlap, chunk_size

load_dotenv()

DOT_FOLDER = Path.home() / ".mozilla" / "firefox"
PROFILES_FILE = "profiles.ini"

TABLE_PLACES = "moz_places"
TABLE_BOOKMARKS = "moz_bookmarks"


def get_profile(dot_folder, profile_name=None):
    """Get the path of the specified profile or the default one.

    If profile_name is None, will return the path to the default profile of the
    user. Otherwise return the path to the selected profile
    """
    config_path = dot_folder / PROFILES_FILE
    config = configparser.ConfigParser()
    config.read(dot_folder / PROFILES_FILE)
    for section in config.sections():
        if (
                (profile_name is None and config[section].get("Default", "") == "1") or
                (profile_name is not None and config[section].get("Name", None) == profile_name)
        ):
            profile_path = Path(config[section].get("Path"))
            if config[section].get("IsRelative") == "1":
                profile_path = dot_folder / profile_path
            return profile_path
    raise RuntimeError("Target profile not found in", config_path)


def get_bookmark_folder_id(con, bookmark_folder_title):
    """get the bookmark folder id from its title."""

    book_res = list(con.execute(f"""
    SELECT id
    FROM {TABLE_BOOKMARKS}
    WHERE title = :title
    LIMIT 1
    """, {"title": bookmark_folder_title}))

    if len(book_res) == 0:
        raise RuntimeError(f"Bookmark folder {bookmark_folder_title} not found")

    book_id = int(book_res[0][0])
    return book_id


@dataclass
class Bookmark:
    """A bookmark content."""

    url: str
    descr: str


def get_bookmarks(con, bookmark_folder_id):
    """Get all bookmarks in the folder with the given id.

    Return a list of tuples (url, descr).
    """
    book_fk = [value[0] for value in con.execute(f"""
    SELECT fk
    FROM {TABLE_BOOKMARKS}
    WHERE parent = :parentid
    """, {"parentid": bookmark_folder_id})]

    bookmarks = list(con.execute(f"""
    SELECT url, title
    FROM {TABLE_PLACES}
    WHERE id in {"(" + ", ".join([str(val) for val in book_fk]) + ")"}
    """))
    return [Bookmark(*bookmark) for bookmark in bookmarks]


def fetch_bookmark_content(bookmark):
    """Fetch page content from bookmark."""

    # langchain loader expects a named file :(
    with (urlopen(bookmark.url) as page,
         NamedTemporaryFile("w+b") as content_file):
        content_file.write(page.read())
        document = UnstructuredHTMLLoader(content_file.name).load()[0]

    # Restore actual URL as source metadata
    document.metadata["source"] = bookmark.url
    # Add page description as metadata
    document.metadata["desc"] = bookmark.descr

    return document


def ingest_into_chroma(db_con, chunks):
    """Ingest document chunks into chroma."""
    # We use the same settings than `ingest.py`
    embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
    collection_name = "langchain"  # default collection name for langchain
    embeddings = SentenceTransformerEmbeddingFunction(embeddings_model_name)

    collection = db_con.get_or_create_collection(
        collection_name,
        embedding_function=embeddings
    )
    collection.add(
        ids=[str(uuid.uuid1()) for _ in chunks],
        documents=[text.page_content for text in chunks],
        metadatas=[text.metadata for text in chunks],
    )
    return collection


def main():
    """Run the bookmark ingestion process."""

    # Read user profile bookmarks
    target_profile = os.environ.get("BOOKMARK_PROFILE", None)
    profile_path = get_profile(DOT_FOLDER, target_profile)
    con = sqlite3.connect(
        f"file:{profile_path / 'places.sqlite'}?immutable=1",
        uri=True
    )

    bookmark_folder = os.environ.get("BOOKMARK_FOLDER")
    if bookmark_folder is None:
        raise RuntimeError("Env variable `BOOKMARK_FOLDER` must be set")
    book_id = get_bookmark_folder_id(con, bookmark_folder)

    bookmarks = get_bookmarks(con, book_id)
    print(f"Found {len(bookmarks)} bookmarks in folder {bookmark_folder}")

    # Fetch content from bookmarks URL
    # TODO try to read it from the browser cache
    documents = [fetch_bookmark_content(bookmark) for bookmark in bookmarks]

    # Split documents and ingest them into chroma
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)

    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    chroma = PersistentClient(persist_directory, settings=CHROMA_SETTINGS)
    collection = ingest_into_chroma(chroma, texts)

    print(f"Added {len(texts)} fragments to collection {collection.name}")


if __name__ == "__main__":
    main()
