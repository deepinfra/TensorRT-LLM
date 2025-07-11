from collections import deque
from typing import Deque, Dict, Optional


class TrieNode:

    def __init__(self) -> None:
        # This is an implementation of Aho-Corasick algorithm
        # Read details from here: https://cp-algorithms.com/string/aho_corasick.html
        self.children: Dict[str, 'TrieNode'] = {}
        self.depth: int = 0
        self.suffix_link: Optional['TrieNode'] = None
        self.word: Optional[str] = None

    def next_node(self, char: str):
        node: Optional['TrieNode'] = self
        while node and char not in node.children:
            node = node.suffix_link
        if node:
            return node.children[char]
        return None

    @staticmethod
    def build_automaton(words: list[str]) -> 'TrieNode':
        root = TrieNode()

        # Build trie
        for word in words:
            node = root
            for char in word:
                node = node.children.setdefault(char, TrieNode())
            node.word = word

        # build suffix links with BFS
        queue: Deque['TrieNode'] = deque()
        for node in root.children.values():
            queue.append(node)
            node.suffix_link = root
            node.depth = root.depth + 1

        while queue:
            current_node = queue.popleft()
            for key, next_node in current_node.children.items():
                queue.append(next_node)
                link = current_node.suffix_link
                # Find the longest proper suffix that is also a prefix
                while link and key not in link.children:
                    link = link.suffix_link
                next_node.suffix_link = link.children[key] if link else root
                next_node.depth = current_node.depth + 1
                next_node.word = next_node.suffix_link.word if next_node.word is None else next_node.word

        return root


class Chunk:

    def __init__(self,
                 text: str,
                 token_ids: list[int],
                 total_tokens: int,
                 output_log_probs: list[float],
                 index: int = 0,
                 finish_reason: Optional[str] = None,
                 stop_reason: Optional[str] = None,
                 done: bool = False,
                 extra=None):
        self.text = text
        self.token_ids = token_ids
        self.total_tokens = total_tokens
        self.output_log_probs = output_log_probs
        self.index = index
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason
        self.done = done
        self.extra = extra

    def __eq__(self, other):
        return (self.text == other.text and self.num_tokens == other.num_tokens
                and self.total_tokens == other.total_tokens)


class PrefixMatchingBufferer:

    def __init__(self, keywords: list[str] | None = None):
        self.root_node: TrieNode = TrieNode.build_automaton(keywords or [])
        self.current_node: TrieNode = self.root_node
        self.buffer: Deque['Chunk'] = deque()
        self.buffered_text_len: int = 0

    def reset(self):
        self.current_node = self.root_node
        self.buffer.clear()
        self.buffered_text_len = 0

    def has_full_match(self) -> bool:
        return self.current_node.word is not None

    def get_match(self) -> str | None:
        return self.current_node.word

    def add_and_release(self, chunk: Chunk) -> list[Chunk]:
        if self.has_full_match():
            return []
        self._add_chunk(chunk)
        return self._release_chunks()

    def release_all(self) -> list[Chunk]:
        released_chunks = []
        while len(self.buffer) > 0:
            released_chunks.append(self.buffer.popleft())
        self.buffered_text_len = 0
        return released_chunks

    def _add_chunk(self, chunk: Chunk) -> None:
        if self.has_full_match():
            # this shouldn't happen, but don't do anything in this case
            return
        chars = []
        for char in chunk.text:
            chars.append(char)
            next_node = self.current_node.next_node(char)
            if next_node is None:
                next_node = self.root_node
            self.current_node = next_node
            if self.has_full_match():
                break
        chunk.text = ''.join(chars)
        self.buffer.append(chunk)
        self.buffered_text_len += len(chunk.text)

    def _release_chunks(self) -> list[Chunk]:
        released_chunks = []

        if self.has_full_match():
            matching_chars_len = len(self.get_match() or '')
        else:
            matching_chars_len = self.current_node.depth
        while len(self.buffer) > 0:
            cur_len = len(self.buffer[0].text)
            if self.buffered_text_len - cur_len >= matching_chars_len:
                released_chunks.append(self.buffer.popleft())
                self.buffered_text_len -= cur_len
            else:
                if self.has_full_match():
                    # if it's a full match, then we need to cut the part
                    chunk = self.buffer[0]
                    prefix_len_to_leave = self.buffered_text_len - matching_chars_len
                    if prefix_len_to_leave > 0:
                        chunk.text = chunk.text[:prefix_len_to_leave]
                        released_chunks.append(chunk)
                        self.buffered_text_len -= len(chunk.text)
                break

        return released_chunks
