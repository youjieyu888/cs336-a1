from __future__ import annotations
import time
from dataclasses import dataclass
from typing import IO, Any, BinaryIO, Generator, Iterator, TypeVar, Generic
from collections.abc import Iterable
import numpy as np
import regex as re
import pickle

PAT_regex = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
UTF8 = 'utf-8'

@dataclass(slots=True)
class DllNode:
     data: bytes = b''
     prev: DllNode|None=None
     next: DllNode|None=None


def connect_dll(node1:DllNode, node2:DllNode):
    node1.next = node2
    node2.prev = node1

def disconnect_dll(node1:DllNode, node2:DllNode):
    node1.next = None
    node2.prev = None

class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = []):
        self._vocab = vocab
        self._merge_rank = {merge: idx for idx, merge in enumerate(merges)}
        self._special_tokens = sorted(special_tokens, key=len, reverse=True)
        self._subword_to_id: dict[bytes, int] = {v: k for k, v in self._vocab.items()}
        for special_token in special_tokens:
            encoded_special_token = special_token.encode(UTF8)
            if encoded_special_token not in self._subword_to_id:
                new_id = len(self._vocab)
                self._vocab[new_id] = encoded_special_token
                self._subword_to_id[encoded_special_token] = new_id
        # --- OPTIMIZATION 1: Pre-compile special tokens regex ---
        if self._special_tokens:
            special_pattern = "|".join(map(re.escape, self._special_tokens))
            self._special_regex = re.compile(f"({special_pattern})")
        else:
            self._special_regex = None # No special tokens to handle


    # def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
    #     pass

    @classmethod
    def from_file(cls, path: str):
        with open(path, 'rb') as f:
            vocab, merges = pickle.load(f)
            return cls(vocab, merges)

    def _get_best_merge_pair(self, head_node: DllNode) -> tuple[DllNode, DllNode] | None:
        """Iterates through the linked list to find the merge with the highest priority (lowest rank)."""
        best_rank = len(self._merge_rank)
        best_pair = None

        node = head_node
        while node and node.next:
            pair = (node.data, node.next.data)
            rank = self._merge_rank.get(pair)
            if rank is not None and rank < best_rank:
                best_rank = rank
                best_pair = (node, node.next)
            node = node.next

        return best_pair

    def tokenize_pretoken(self, text: bytes) -> list[int]:
        """
        Encodes a byte string using a doubly-linked list for efficient merging.
        This is the new, high-performance version.
        """
        if not text:
            return []
        if len(text) == 1:
            return [self._subword_to_id[text]]

        # 1. Build the doubly-linked list from bytes
        head = DllNode(text[0:1])
        current = head
        for i in range(1, len(text)):
            new_node = DllNode(text[i:i + 1], prev=current)
            current.next = new_node
            current = new_node

        # 2. Repeatedly find and perform the best merge
        while True:
            pair_to_merge = self._get_best_merge_pair(head)

            if pair_to_merge is None:
                break  # No more possible merges

            node1, node2 = pair_to_merge

            # Merge data
            node1.data += node2.data

            # Remove node2 by updating pointers
            node1.next = node2.next
            if node2.next:
                node2.next.prev = node1

        # 3. Convert final linked list to token IDs
        ids = []
        node = head
        while node:
            ids.append(self._subword_to_id[node.data])
            node = node.next

        return ids

    def _split_into_pretokens(self, text: str) -> list[str | int]:
        if self._special_regex is None:
            return PAT_regex.findall(text)
        result = []
        last_end = 0
        for m in self._special_regex.finditer(text):
            # normal text before the token
            if m.start() > last_end:
                result.extend(PAT_regex.findall(text[last_end:m.start()]))
            # transformed special token
            encoded_special_token = text[m.start():m.end()].encode(UTF8)
            result.append(self._subword_to_id[encoded_special_token])
            last_end = m.end()

        # tail after last match
        if last_end < len(text):
            result.extend(PAT_regex.findall(text[last_end:]))
        return result

    def encode(self, text: str) -> list[int]:
        chunks = self._split_into_pretokens(text)
        return self.encode_chunks(chunks)

    # int input means special token already encoded, str means pretoken
    def encode_chunks(self, chunks: list[str|int])->list[int]:
        tokenization = []
        for chunk in chunks:
            if type(chunk)==int:
                tokenization.append(chunk)
            else:
                tokenization.extend(self.tokenize_pretoken(chunk.encode(UTF8)))
        return tokenization

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        leftover = ''
        processed_chars=0
        st_time = time.time()
        max_leftover_len = 0
        for text in iterable:
            text = leftover + text
            chunks = self._split_into_pretokens(text)
            if type(chunks[-1]) == str:
                yield from self.encode_chunks(chunks[:-1])
                leftover = chunks[-1]
            else:
                yield from self.encode_chunks(chunks)
                leftover = ''
            processed_chars += len(text)
            print('leftover:===================\n',leftover)
            print(f'processed {processed_chars} chars in {time.time()-st_time}s')
            st_time = time.time()
            processed_chars=0
        yield from self.encode_chunks(self._split_into_pretokens(leftover))

    def decode(self, ids: list[int]) -> str:
        bytes_list = []
        for id in ids:
            bytes_list.append(self._vocab[id])
        return b''.join(bytes_list).decode(UTF8, errors='replace')


