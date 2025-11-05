import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

################### JSON-like TXT FILE HANDLING ###################
def preprocess_text(text: str) -> str:
    url_pattern = re.compile(r'"url"\s*:\s*".*?"\s*,?|\"url\"\s*:\s*null\s*,?')
    text = re.sub(url_pattern, "", text)

    related_links_pattern = re.compile(
        r'"related_links"\s*:\s*(null|\[.*?\]|\{.*?\})\s*,?',
        flags=re.DOTALL
    )
    text = re.sub(related_links_pattern, "", text)

    dots_pattern = re.compile(r"\.{2,}")
    text = re.sub(dots_pattern, ".", text)

    return text.strip()

def split_into_blocks(text: str):
    blocks = text.split(',  "title":')
    return [b.strip() for b in blocks if b.strip()]

def normalize_text(s: str) -> str:
    s = s.replace('\\n', ' ')
    s = re.sub(r'\.{2,}', '.', s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\s*\n\s*', ' ', s)
    return s.strip()

def clean_block(block: str, is_first: bool = False) -> dict:
    b = block.strip()

    if is_first and '"snippet":' not in b and '"title":' not in b:
        return {"intro_snippet": normalize_text(b)}

    m_title_key = re.search(r'"title"\s*:\s*"([^"]+)"', b, flags=re.DOTALL)
    if m_title_key:
        title = m_title_key.group(1).strip()
        m_snip = re.search(r'"snippet"\s*:\s*"([\s\S]*?)"\s*(?:,|$)', b, flags=re.DOTALL)
        snippet = m_snip.group(1).strip() if m_snip else ''
        return {"title": normalize_text(title), "snippet": normalize_text(snippet)}

    m = re.match(r'^"?(?P<title>[^"]+?)"?\s*\}\s*,\s*\{\s*"snippet"\s*:\s*"(.*)$',
                 b, flags=re.DOTALL)
    if m:
        title = m.group('title').strip()
        snippet_rest = m.group(2)
        snippet_rest = re.sub(r'"\s*$|\}\s*,\s*$|\}\s*$|",\s*$', '', snippet_rest).strip()
        return {"title": normalize_text(title), "snippet": normalize_text(snippet_rest)}

    m2 = re.match(r'^(?P<title>.+?)\}\s*,\s*\{\s*(?P<rest>.*)$', b, flags=re.DOTALL)
    if m2:
        title = m2.group('title').strip().strip('"')
        rest = m2.group('rest').strip()
        m_snip2 = re.search(r'"snippet"\s*:\s*"([\s\S]*?)"\s*(?:,|$)', rest, flags=re.DOTALL)
        snippet = m_snip2.group(1).strip() if m_snip2 else rest
        return {"title": normalize_text(title), "snippet": normalize_text(snippet)}

    return {"raw_block": normalize_text(b)}

def process_file(text: str):
    cleaned = preprocess_text(text)
    blocks = split_into_blocks(cleaned)

    structured = []
    for i, block in enumerate(blocks):
        structured.append(clean_block(block, is_first=(i==0)))
    return structured

def create_chunks_from_jsonlike(structured: list):
    chunks = []

    # handle intro_snippet
    intro_text = ""
    if structured and "intro_snippet" in structured[0]:
        intro_text = structured[0]["intro_snippet"]
        structured = structured[1:]  # remove intro block

    # pairwise combine title+snippet
    for i in range(0, len(structured), 2):
        first = structured[i]
        second = structured[i+1] if i+1 < len(structured) else None

        text_parts = []
        if intro_text and i == 0:  # concat intro with first pair
            text_parts.append(intro_text)

        text_parts.append(first.get("title", "") + " " + first.get("snippet", ""))

        if second:
            text_parts.append(second.get("title", "") + " " + second.get("snippet", ""))

        chunks.append(" ".join(text_parts).strip())

    return chunks


################### NORMAL TXT FILE HANDLING ###################
def chunk_text(text: str, chunk_size: int = 100, chunk_overlap: int = 20):
    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 6,
        chunk_overlap=chunk_overlap * 6,
        separators=separators,
        length_function=len
    )
    return splitter.split_text(text)


################### MAIN PIPELINE ###################
def process_txt_file(filename: str, text: str):
    if re.search(r'"url"|"snippet"', text):
        structured = process_file(text)
        chunks = create_chunks_from_jsonlike(structured)
        return {"type": "json_like", "chunks": chunks}
    else:
        chunks = chunk_text(text)
        return {"type": "normal", "chunks": chunks}

def build_corpus(directory: str):
    corpus = {}
    for fname in os.listdir(directory):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(directory, fname), "r", encoding="utf-8") as f:
            text = f.read()

        result = process_txt_file(fname, text)
        chunks = result["chunks"]
        
        base_name = os.path.splitext(fname)[0]
        for i, chunk in enumerate(chunks, 1):
            chunk_id = f"{base_name}_chunk{i}"
            corpus[chunk_id] = {
                "document": fname,
                "text": chunk
            }
    return corpus

corpus = build_corpus("./mock_dataset_v1")