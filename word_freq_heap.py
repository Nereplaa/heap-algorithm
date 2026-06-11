"""
word_freq_heap.py
==================

Counts word frequencies in a TXT file and prints them sorted:

    1. Alphabetically by the FIRST LETTER of the word (A -> Z), then
    2. For words that share the same first letter, by FREQUENCY,
       descending (highest count first).

The entire ordering above is produced by ONE hand-written binary heap
(class `DualKeyHeap` below), ordered by exactly the two keys required
by the assignment:

    PRIMARY KEY   : first letter of the word   -> ascending  (A < B < ... < Z)
    SECONDARY KEY : occurrence count of word    -> descending (used ONLY to
                                                    break ties between words
                                                    that share the same
                                                    first letter)

No `heapq`, `sorted()`, `PriorityQueue`, or any other built-in
priority-queue / sorting utility is used to build or order the heap.
The heap is a plain Python list used as an array-based binary tree,
exactly like the classic textbook heap.

For the full write-up (algorithm explanation, complexity analysis,
worked example with the Kocaeli/Konya/... input, and edge cases) see
REPORT.md.

USAGE
-----
    python word_freq_heap.py <path_to_file.txt>            normal run
    python word_freq_heap.py <path_to_file.txt> --debug    show heap state
                                                            after every word
    python word_freq_heap.py --demo                        run the built-in
                                                            Kocaeli/Konya/...
                                                            walkthrough from
                                                            the assignment
    python word_freq_heap.py                               prompts for a
                                                            file path
"""

from __future__ import annotations

import argparse
import re
import sys


# ======================================================================
#  WordNode
# ======================================================================
class WordNode:
    """
    One element stored in the heap: a normalized word and how many
    times it has been seen so far.

    Using a tiny class (instead of a bare list/tuple) makes the heap
    code below read like "node.word" / "node.count" instead of
    "node[0]" / "node[1]", which is much easier to follow.
    """

    __slots__ = ("word", "count")

    def __init__(self, word: str, count: int = 1) -> None:
        self.word = word
        self.count = count

    def __repr__(self) -> str:
        return f"{self.word}({self.count})"


# ======================================================================
#  DualKeyHeap  -- the custom heap required by the assignment
# ======================================================================
class DualKeyHeap:
    """
    A binary heap, stored as a plain Python list (array-based, exactly
    like the classic textbook heap: the children of index ``i`` live
    at ``2*i + 1`` and ``2*i + 2``, and the parent of ``i`` lives at
    ``(i - 1) // 2``).

    ORDERING RULE ("dual key")
    ---------------------------
    The node that sits closer to the root is the one with HIGHER
    PRIORITY, defined as:

        1. The node whose word's FIRST LETTER is alphabetically
           smaller (A beats B beats C ... beats Z).
        2. If both nodes start with the SAME letter, the node with the
           LARGER occurrence COUNT wins.

    Because of this rule, repeatedly removing the root of the heap
    (``extract_top``) yields nodes in EXACTLY the order the assignment
    asks for as final output: A -> Z, and for ties on the first
    letter, highest count -> lowest count.

    FAST "ALREADY EXISTS?" LOOKUP
    ------------------------------
    A plain heap can only tell you about its root in O(1); finding an
    arbitrary word would normally take O(n). To satisfy the
    requirement "before inserting a word, check whether it already
    exists, and if so increment + re-heapify", this heap keeps a side
    dictionary ``self._index`` mapping ``word -> position in the
    array``. Every time two nodes are swapped, both of their entries
    in ``self._index`` are updated, so the dictionary always reflects
    the true position of every word. This turns the existence check
    and the "find the node to update" step into O(1) operations.
    """

    def __init__(self) -> None:
        # The heap itself: array-based binary tree of WordNode objects.
        self._data: list[WordNode] = []

        # word -> index of that word's node inside self._data.
        # Kept in sync on every swap so we can find any word in O(1).
        self._index: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Basic introspection
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._data)

    def is_empty(self) -> bool:
        """Return True if the heap currently holds no words."""
        return len(self._data) == 0

    def contains(self, word: str) -> bool:
        """Return True if ``word`` already has a node in the heap."""
        return word in self._index

    def to_list(self) -> list[tuple[str, int]]:
        """
        Return a snapshot of the heap's underlying array as a list of
        ``(word, count)`` tuples, in array order (index 0 = root).
        Useful for debugging / printing.
        """
        return [(node.word, node.count) for node in self._data]

    # ------------------------------------------------------------------
    # The dual-key comparison: this single method encodes BOTH keys.
    # ------------------------------------------------------------------
    def _has_priority(self, i: int, j: int) -> bool:
        """
        Return True if the node at index ``i`` has STRICTLY HIGHER
        priority than the node at index ``j``, i.e. ``i`` is allowed
        to sit closer to the root than ``j``.

        Dual-key rule:
          - PRIMARY:   smaller first letter  -> higher priority
          - SECONDARY: (only if first letters are equal) larger count
                       -> higher priority
        """
        a, b = self._data[i], self._data[j]

        letter_a, letter_b = a.word[0], b.word[0]
        if letter_a != letter_b:
            # Primary key: alphabetical order, A is "smallest" (root-most).
            return letter_a < letter_b

        # Same first letter -> secondary key decides: higher count wins.
        return a.count > b.count

    # ------------------------------------------------------------------
    # Internal helpers: swap + sift (a.k.a. heapify) up / down
    # ------------------------------------------------------------------
    def _swap(self, i: int, j: int) -> None:
        """Swap two nodes in the array AND keep self._index in sync."""
        self._data[i], self._data[j] = self._data[j], self._data[i]
        self._index[self._data[i].word] = i
        self._index[self._data[j].word] = j

    def _sift_up(self, i: int) -> None:
        """
        Move the node at index ``i`` UP the tree while it has higher
        priority than its parent. This is the standard
        "bubble up" step used both for brand-new nodes (appended at
        the end of the array) and for existing nodes whose count just
        increased.
        """
        while i > 0:
            parent = (i - 1) // 2
            if self._has_priority(i, parent):
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        """
        Move the node at index ``i`` DOWN the tree, repeatedly swapping
        it with whichever child has the higher priority, until both of
        its children have lower (or equal) priority. Used after the
        root is removed and replaced by the last element in the array.
        """
        n = len(self._data)
        while True:
            best = i
            left, right = 2 * i + 1, 2 * i + 2

            if left < n and self._has_priority(left, best):
                best = left
            if right < n and self._has_priority(right, best):
                best = right

            if best == i:
                break

            self._swap(i, best)
            i = best

    # ------------------------------------------------------------------
    # Public operation #1: insert a new word OR bump an existing one
    # ------------------------------------------------------------------
    def insert_or_increment(self, word: str) -> None:
        """
        Process one word read from the file.

          * If the word is ALREADY in the heap: increment its count
            and re-heapify (sift up) so the heap property is restored.
          * If the word is NOT yet in the heap: insert it as a new
            leaf with count = 1, then sift it up into place.

        NOTE on why only `_sift_up` is needed for an increment:
        Increasing a node's count can never DECREASE its priority
        (the dual-key rule only ever compares counts between nodes
        that already share the same first letter, and a higher count
        is always >= as good as a lower one). Since the heap was valid
        *before* the increment, the only possible violation afterwards
        is "this node now deserves to be HIGHER than its parent" -
        which is exactly what `_sift_up` fixes. A `_sift_down` can
        never be required.
        """
        if word in self._index:
            i = self._index[word]
            self._data[i].count += 1
            self._sift_up(i)
        else:
            self._data.append(WordNode(word, 1))
            i = len(self._data) - 1
            self._index[word] = i
            self._sift_up(i)

    # ------------------------------------------------------------------
    # Public operation #2: remove and return the highest-priority node
    # ------------------------------------------------------------------
    def extract_top(self) -> WordNode:
        """
        Remove and return the node currently at the root (the node
        with the highest priority under the dual-key rule).

        Repeatedly calling this until the heap is empty yields nodes
        in exactly the order required by the assignment:
            A -> Z by first letter, and for ties on the first letter,
            highest count -> lowest count.
        """
        if not self._data:
            raise IndexError("extract_top() called on an empty heap")

        top = self._data[0]
        last = self._data.pop()
        del self._index[top.word]

        if self._data:
            self._data[0] = last
            self._index[last.word] = 0
            self._sift_down(0)

        return top

    # ------------------------------------------------------------------
    # Debug / visualization helpers
    # ------------------------------------------------------------------
    def print_array(self) -> None:
        """Print the heap's underlying array, e.g. [Adana(2), Konya(2), ...]."""
        print("    array: [" + ", ".join(repr(n) for n in self._data) + "]")

    def print_tree(self) -> None:
        """
        Pretty-print the heap as an ASCII tree, root at the top,
        children indented underneath with branch connectors.

        Plain ASCII characters ("+--", "`--", "|") are used (instead
        of Unicode box-drawing characters) so the output displays
        correctly in any terminal/console encoding.
        """
        if not self._data:
            print("    (empty heap)")
            return
        print("    " + repr(self._data[0]))
        self._print_children(0, "    ")

    def _print_children(self, i: int, prefix: str) -> None:
        n = len(self._data)
        kids = [c for c in (2 * i + 1, 2 * i + 2) if c < n]
        for idx, child in enumerate(kids):
            is_last = idx == len(kids) - 1
            branch = "`-- " if is_last else "+-- "
            print(prefix + branch + repr(self._data[child]))
            extension = "    " if is_last else "|   "
            self._print_children(child, prefix + extension)


# ======================================================================
#  Text processing helpers
# ======================================================================
def extract_words(text: str) -> list[str]:
    """
    Turn raw file contents into a list of normalized words, in the
    order they appear.

    Normalization rules:
      - Everything is lower-cased, so "Ankara", "ankara" and "ANKARA"
        all become the same word.
      - Punctuation, digits, whitespace and any other non-letter
        characters are treated as separators and discarded: a "word"
        is defined as a maximal run of Unicode letter characters.
        e.g. "Adana," / "(adana)" / "adana!" / "ADANA" all become
        "adana"; "word2vec" becomes the two words "word" and "vec".
    """
    return re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)


# ======================================================================
#  Driver: build the heap from a word stream, with optional debug trace
# ======================================================================
def build_heap(words: list[str], debug: bool = False) -> DualKeyHeap:
    """
    Feed every word into a fresh DualKeyHeap, one at a time, exactly as
    described by the assignment ("the heap must be updated after
    reading EVERY word from the file").

    If ``debug`` is True, the heap's array and tree representation are
    printed after each word is processed, so the step-by-step
    behaviour can be inspected.
    """
    heap = DualKeyHeap()

    for step, word in enumerate(words, start=1):
        heap.insert_or_increment(word)

        if debug:
            print(f"Step {step}: read '{word}'")
            heap.print_array()
            heap.print_tree()
            print()

    return heap


def print_results(heap: DualKeyHeap) -> None:
    """
    Drain the heap with repeated extract_top() calls and print a
    formatted (word, count) table. Because of the dual-key heap
    ordering, the words come out already sorted A->Z, and (for equal
    first letters) highest count -> lowest count - no extra sorting
    step is needed.
    """
    if heap.is_empty():
        print("(no words found)")
        return

    print(f"{'Word':<20}{'Count':>6}")
    print("-" * 26)
    while not heap.is_empty():
        node = heap.extract_top()
        print(f"{node.word:<20}{node.count:>6}")


# ======================================================================
#  Demo mode: the worked example from the assignment description
# ======================================================================
DEMO_WORDS = ["Kocaeli", "Konya", "Van", "Ankara", "Konya", "Sivas", "Adana", "Adana"]


def run_demo() -> None:
    """
    Run the exact worked example given in the assignment:

        Kocaeli, Konya, Van, Ankara, Konya, Sivas, Adana, Adana

    printing the heap's array + tree representation after every single
    word, then the final sorted result.
    """
    print("=" * 60)
    print("DEMO: dual-key heap, step by step")
    print("Input words:", ", ".join(DEMO_WORDS))
    print("=" * 60)
    print()

    words = [w.lower() for w in DEMO_WORDS]
    heap = build_heap(words, debug=True)

    print("=" * 60)
    print("FINAL RESULT (extracted from the heap, A->Z, count desc.)")
    print("=" * 60)
    print_results(heap)


# ======================================================================
#  Main entry point
# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count word frequencies in a TXT file and print them sorted "
            "A->Z (by first letter), then by frequency descending for "
            "words sharing the same first letter. Sorting is performed "
            "entirely by a custom dual-key heap."
        )
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="path to the input .txt file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print the heap's array and tree state after every word",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the built-in Kocaeli/Konya/Van/... example from the "
        "assignment instead of reading a file",
    )
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    filepath = args.file
    if not filepath:
        filepath = input("Enter path to a .txt file: ").strip()

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        print(f"Error: could not open '{filepath}': {exc}")
        sys.exit(1)

    words = extract_words(text)

    if not words:
        print("The file contains no words.")
        return

    heap = build_heap(words, debug=args.debug)

    if args.debug:
        print("=" * 60)
        print("FINAL RESULT")
        print("=" * 60)

    print_results(heap)


if __name__ == "__main__":
    main()
