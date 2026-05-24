import math
import re
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
INDEX_PATH = Path(__file__).resolve().parents[2] / ".kb" / "index.json"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "is",
    "it",
    "my",
    "of",
    "the",
    "to",
    "what",
    "when",
    "which",
}


@dataclass
class Section:
    id: str
    file: str
    heading: str
    heading_path: list[str]
    content: str
    tokens: list[str]
    metadata: dict = field(default_factory=dict)  # YAML frontmatter (future use)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "content": self.content,
            "tokens": self.tokens,
            "metadata": self.metadata,
        }


sections: list[Section] = []
doc_freq: Counter[str] = Counter()
avg_doc_len = 0.0
files_indexed = 0


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


def parse_markdown(path: Path) -> list[Section]:
    """Parse one Markdown file into Sections under the body-bearing rule.

    See CONTEXT.md > Section for the formal definition. The 10-rule spec:

    1.  Read the file as UTF-8.
    2.  If the file starts with `---\\n`, strip and parse the YAML frontmatter
        into a dict. Attach this dict to every Section's `metadata` field.
        Do NOT tokenize frontmatter values into BM25 tokens.
    3.  Scan the remaining body line by line, maintaining `in_fence: bool`.
        Toggle `in_fence` whenever a line starts with three backticks. While
        `in_fence` is true, treat every line as content — do NOT match
        HEADING_RE against fenced code (so `# bash comment` inside a code
        block is not treated as a heading).
    4.  Outside fences, match HEADING_RE. Use a stack to track the current
        heading path. When a heading at depth d arrives, pop the stack until
        the top has depth < d (those headings are "closed" and emitted as
        Sections if they qualify under rule 5). Then push the new heading.
    5.  A heading becomes a Section when either:
            (i)  It is a leaf — from its push to its pop, no deeper heading
                 was ever pushed on top of it; OR
            (ii) It is body-bearing — between its push and the first deeper
                 heading pushed on top of it, the body content accumulated
                 directly under it is not whitespace-only.
        In case (ii) the Section's content is only the body owned directly
        by this heading, NOT the recursive content of its children.
    6.  Emit a `log_event("parse_warning", ...)` whenever a non-leaf heading
        has only whitespace body and therefore produces no Section (this is
        normal for h1 file titles, but worth logging at startup).
    7.  A Source with zero headings produces a single Section: `id=filename`
        (no `#anchor`), `heading=filename`, `heading_path=[filename]`,
        `content=` full file body.
    8.  An empty-body leaf (heading present, body whitespace-only) is still
        emitted as a Section. Its `content` is `""`; its `tokens` come from
        the heading text alone. BM25 will rank it low unless the query
        matches the heading directly, which is the desired behavior.
    9.  Heading slug collisions inside the same Source: append `-2`, `-3`, …
        suffixes. Never silently overwrite a previously emitted Section.
    10. `tokens` is the concatenation of (a) lowercase alphanumeric tokens
        from the heading text and (b) the same for the body content, with
        STOP_WORDS removed. The same tokenization applies to query strings
        at retrieval time.
    """
    return []


def write_index_json(index_path: Path = INDEX_PATH) -> None:
    # TODO: Persist the section index to .kb/index.json so it is inspectable.
    #
    # Hints:
    # 1. Create index_path.parent if it does not exist.
    # 2. Write {"sections": [...], "stats": {...}} as pretty JSON.
    # 3. Use section.to_dict() for each Section.
    _ = json


def rebuild_stats() -> None:
    # TODO: Rebuild doc_freq, avg_doc_len, and files_indexed from sections.
    #
    # Hints:
    # 1. files_indexed can be derived from the unique section.file values.
    # 2. doc_freq counts how many sections contain each token.
    # 3. avg_doc_len is the average token count across sections.
    pass


def load_index_json(index_path: Path = INDEX_PATH) -> tuple[int, int]:
    # TODO: Load .kb/index.json into the in-memory sections list.
    #
    # Hints:
    # 1. If index_path does not exist, return (0, 0).
    # 2. Read payload["sections"] and convert each item back to Section.
    # 3. Call rebuild_stats() after assigning sections.
    # 4. Return (files_indexed, sections_indexed).
    return 0, 0


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    global sections, doc_freq, avg_doc_len, files_indexed

    # TODO: Build an in-memory section index from docs/*.md.
    #
    # Hints:
    # 1. Read all Markdown files from docs_dir.
    # 2. Call parse_markdown() for each file.
    # 3. Call rebuild_stats() to compute BM25 metadata.
    # 4. Persist .kb/index.json with write_index_json().
    # 5. Call write_index_json() so students can inspect the generated index.
    # 6. Return (files_indexed, sections_indexed).
    sections = []
    doc_freq = Counter()
    avg_doc_len = 0.0
    files_indexed = 0
    write_index_json()
    return files_indexed, len(sections)


def bm25_score(query_tokens: list[str], section: Section, k1: float = 1.5, b: float = 0.75) -> float:
    # TODO: Score one section for the query using BM25.
    #
    # Hints:
    # 1. Count term frequency in the section.
    # 2. Use doc_freq to give rare terms higher weight.
    # 3. Normalize by section length using avg_doc_len.
    # 4. Add a small boost when query terms appear in heading_path.
    return 0.0


def search(query: str, k: int = 3) -> list[tuple[Section, float]]:
    query_tokens = tokenize(query)
    ranked = [
        (section, bm25_score(query_tokens, section))
        for section in sections
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [(section, score) for section, score in ranked[:k] if score > 0]
