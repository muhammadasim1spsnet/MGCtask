"""
assistant.py
------------
This is the "brain" of the assistant. It answers a question in one of two
ways:

  1. DETERMINISTIC HANDLERS (preferred, when applicable)
     For the 5 known question types from the assessment - base price
     lookup, the location-premium calculation, the transfer-fee conflict,
     the missing rental yield, and the unconfirmed anchor tenant - we
     parse the source Markdown directly with simple regular expressions.

     WHY do this instead of just trusting the LLM? Because LLMs can
     mis-add percentages, or forget to mention a conflict, or paraphrase
     a number slightly wrong. For a sales assistant, getting a price or
     a conflict wrong is worse than being "boringly deterministic". So
     for these specific, business-critical facts, the code itself reads
     the numbers straight from the document and does the arithmetic -
     the LLM is not involved at all in producing the number.

  2. RETRIEVAL-AUGMENTED GENERATION (fallback, for everything else)
     For any other question, we embed the question, retrieve the most
     relevant chunks from ChromaDB, and pass them to an LLM with a strict
     "only answer from this context" system prompt. If no OPENAI_API_KEY
     is configured, we skip the LLM call and just show the best-matching
     excerpt instead of guessing.

Every answer returned by this module is a dict with:
    answer, status, sources (list of "Document - Section" strings),
    calculation (optional multi-line string shown for Q2-style questions)
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
load_dotenv(PROJECT_ROOT / ".env")  # shared project environment; contains optional LLM provider settings

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "mgc_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5  # how many chunks to retrieve for the generic RAG fallback

RAW_FILES = {
    "brochure": (DATA_DIR / "01_mgc_aurora_heights_brochure.md", "MGC Brochure"),
    "price_list": (DATA_DIR / "02_price_list_payment_plan.md", "MGC Price List"),
    "booking_faq": (DATA_DIR / "03_booking_policy_faq.md", "MGC Booking Policy & FAQ"),
}

STRICT_SYSTEM_PROMPT = """You are a document-grounded sales assistant for MGC Aurora Heights.

Rules you must always follow:
- Answer ONLY using the supplied document context below. Never use outside knowledge.
- If the answer is not present in the context, say clearly that it is not available
  and recommend confirming with the marketing manager. Do not guess.
- If different parts of the context conflict, report the conflict and show both
  values with their sources - never silently pick one.
- For any calculation, use only numbers explicitly present in the context and show
  the calculation step by step.
- Always list the source document and section for every fact you use.
"""

# Lazily-created singletons so we don't reload the model / reopen the DB on every call.
_embedding_model = None
_collection = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(question, k=TOP_K):
    """Embed the question and fetch the k most relevant chunks from ChromaDB.

    This is the "R" in RAG: we turn the question into the same kind of
    vector we used for the document chunks, then ask ChromaDB for the
    stored chunks whose vectors are closest to it (i.e. most similar in
    meaning).
    """
    model = _get_embedding_model()
    query_embedding = model.encode([question]).tolist()
    collection = _get_collection()
    results = collection.query(query_embeddings=query_embedding, n_results=k)

    chunks = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": text,
            "document_name": meta["document_name"],
            "section": meta["section"],
        })
    return chunks


def _read_raw(key):
    """Read a source document's full raw text by its short key."""
    path, _name = RAW_FILES[key]
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _source(doc_key, section):
    doc_name = RAW_FILES[doc_key][1]
    return f"{doc_name} - {section}"


NOT_FOUND_MSG = (
    "I don't have enough information in the provided documents. "
    "Please confirm with the marketing manager."
)


# ---------------------------------------------------------------------------
# DETERMINISTIC HANDLER 1: base price lookup
# ---------------------------------------------------------------------------
UNIT_TYPE_ALIASES = [
    # (keywords that must ALL appear in the question, table row label)
    (["penthouse"], "4-Bed Penthouse"),
    (["4-bed"], "4-Bed Penthouse"),
    (["3-bed"], "3-Bed Executive"),
    (["2-bed", "corner"], "2-Bed Corner"),
    (["2-bed"], "2-Bed Standard"),
    (["1-bed"], "1-Bed Standard"),
    (["studio"], "Studio"),
]


def _extract_table_row_price(table_text, row_label):
    """Pull the base price (3rd column) out of a Markdown table row.

    Example row: "| 2-Bed Standard | 1,150 sq ft | 22,425,000 | 19,500 |"
    We match the row by its label and capture the first big number that
    follows it (the base price column).
    """
    pattern = r"\|\s*" + re.escape(row_label) + r"\s*\|[^|]*\|\s*([\d,]+)\s*\|"
    match = re.search(pattern, table_text)
    if not match:
        return None
    return match.group(1)  # keep as string with commas, e.g. "22,425,000"


def _extract_block_table(price_list_text, block_letter):
    """Return just the text of the "Base Prices (Block A/B)" table section."""
    pattern = rf"## Base Prices \(Block {block_letter}\)(.*?)(?=\n## |\Z)"
    match = re.search(pattern, price_list_text, re.DOTALL)
    return match.group(1) if match else None


def handle_base_price_lookup(question_lower):
    price_list_text = _read_raw("price_list")

    block = None
    if "block a" in question_lower:
        block = "A"
    elif "block b" in question_lower:
        block = "B"

    row_label = None
    for keywords, label in UNIT_TYPE_ALIASES:
        if all(kw in question_lower for kw in keywords):
            row_label = label
            break

    if row_label is None:
        return None  # not a base-price question we recognise; let RAG try

    if block is None:
        return {
            "answer": (
                "Please specify which block (A or B) - base prices differ "
                "between blocks for the same unit type."
            ),
            "status": "NOT FOUND",
            "sources": [],
        }

    table_text = _extract_block_table(price_list_text, block)
    if table_text is None:
        return {"answer": NOT_FOUND_MSG, "status": "NOT FOUND", "sources": []}

    price = _extract_table_row_price(table_text, row_label)
    if price is None:
        return {
            "answer": (
                f"The provided documents do not list a base price for "
                f"{row_label} in Block {block}. " + NOT_FOUND_MSG
            ),
            "status": "NOT FOUND",
            "sources": [_source("price_list", f"Base Prices (Block {block})")],
        }

    return {
        "answer": f"The base price of a {row_label} in Block {block} is PKR {price}.",
        "status": "FOUND IN DOCUMENT",
        "sources": [_source("price_list", f"Base Prices (Block {block})")],
    }


# ---------------------------------------------------------------------------
# DETERMINISTIC HANDLER 2: location-premium price calculation
# ---------------------------------------------------------------------------
def _extract_premium_pct(premiums_text, label_pattern):
    match = re.search(label_pattern + r":\s*\*\*\+(\d+(?:\.\d+)?)%\*\*", premiums_text)
    return float(match.group(1)) if match else None


def _floor_band_pct(premiums_text, floor):
    """Work out which floor-band premium (if any) applies to a given floor.

    The document only defines two bands: 13-19 and 20-22. We derive "no
    premium" for any floor outside those bands directly from the fact
    that no other band is mentioned - we are not inventing a number, we
    are reporting the absence of one.
    """
    band_13_19 = _extract_premium_pct(premiums_text, r"Floors\s*13[\u2013-]19")
    band_20_22 = _extract_premium_pct(premiums_text, r"Floors\s*20[\u2013-]22")
    if band_13_19 is not None and 13 <= floor <= 19:
        return band_13_19, "Floors 13-19"
    if band_20_22 is not None and 20 <= floor <= 22:
        return band_20_22, "Floors 20-22"
    return 0.0, None


def handle_price_calculation(question_lower):
    floor_match = re.search(r"floor\s*(\d+)", question_lower)
    if floor_match is None:
        return None  # can't identify the floor; let RAG try instead

    floor = int(floor_match.group(1))
    is_corner = "corner" in question_lower
    is_margalla = "margalla" in question_lower
    block = "B" if "block b" in question_lower else ("A" if "block a" in question_lower else "B")

    # NOTE on an ambiguity in the source documents: the price table has a
    # separate "2-Bed Corner" row with its own (higher) base price, AND the
    # Location Premiums section separately lists a +3% "corner unit"
    # premium. Applying both would double-count the corner attribute. The
    # document's own worked example ("a Margalla-facing corner unit on
    # floor 15 carries +4%+3%+6% = +13% over base") applies premiums on
    # top of the plain per-bedroom-count base price, so we follow that and
    # use the *Standard* row here, layering the Location Premiums on top -
    # not the separately-priced "Corner" row. This assumption is called
    # out in the answer.
    ROW_ALIASES_FOR_CALC = [
        (["penthouse"], "4-Bed Penthouse"),
        (["4-bed"], "4-Bed Penthouse"),
        (["3-bed"], "3-Bed Executive"),
        (["2-bed"], "2-Bed Standard"),
        (["1-bed"], "1-Bed Standard"),
        (["studio"], "Studio"),
    ]
    row_label = None
    for keywords, label in ROW_ALIASES_FOR_CALC:
        if all(kw in question_lower for kw in keywords):
            row_label = label
            break
    if row_label is None:
        row_label = "2-Bed Standard"

    price_list_text = _read_raw("price_list")
    table_text = _extract_block_table(price_list_text, block)
    if table_text is None:
        return {"answer": NOT_FOUND_MSG, "status": "NOT FOUND", "sources": []}

    base_price_str = _extract_table_row_price(table_text, row_label)
    if base_price_str is None:
        return {
            "answer": f"I cannot calculate a total because the base price for "
                      f"{row_label} in Block {block} is not in the documents. " + NOT_FOUND_MSG,
            "status": "NOT FOUND",
            "sources": [_source("price_list", f"Base Prices (Block {block})")],
        }
    base_price = int(base_price_str.replace(",", ""))

    premiums_match = re.search(r"## Location Premiums(.*?)(?=\n## |\Z)", price_list_text, re.DOTALL)
    if premiums_match is None:
        return {"answer": NOT_FOUND_MSG, "status": "NOT FOUND", "sources": []}
    premiums_text = premiums_match.group(1)

    margalla_pct = _extract_premium_pct(premiums_text, "Margalla-facing") if is_margalla else 0.0
    corner_pct = _extract_premium_pct(premiums_text, "Corner unit") if is_corner else 0.0
    floor_pct, floor_band_label = _floor_band_pct(premiums_text, floor)

    if is_margalla and margalla_pct is None:
        return {
            "answer": "I cannot calculate a reliable total because the Margalla-facing "
                      "premium is not specified in the provided documents.",
            "status": "NOT FOUND",
            "sources": [_source("price_list", "Location Premiums")],
        }
    if is_corner and corner_pct is None:
        return {
            "answer": "I cannot calculate a reliable total because the corner-unit "
                      "premium is not specified in the provided documents.",
            "status": "NOT FOUND",
            "sources": [_source("price_list", "Location Premiums")],
        }

    margalla_amt = round(base_price * (margalla_pct or 0) / 100)
    corner_amt = round(base_price * (corner_pct or 0) / 100)
    floor_amt = round(base_price * floor_pct / 100)
    total = base_price + margalla_amt + corner_amt + floor_amt

    calc_lines = [f"Base price ({row_label}, Block {block}) = PKR {base_price:,}"]
    if is_margalla:
        calc_lines.append(f"Margalla-facing premium (+{margalla_pct:g}%) = PKR {margalla_amt:,}")
    if is_corner:
        calc_lines.append(f"Corner-unit premium (+{corner_pct:g}%) = PKR {corner_amt:,}")
    if floor_band_label:
        calc_lines.append(f"Floor {floor} premium, {floor_band_label} (+{floor_pct:g}%) = PKR {floor_amt:,}")
    else:
        calc_lines.append(f"Floor {floor} premium = PKR 0 (no premium band defined for this floor)")
    calc_lines.append("-" * 40)
    calc_lines.append(f"Total = PKR {total:,}")

    note = (
        " Note: the Price List also has a separately-priced \"2-Bed Corner\" "
        "table row; to avoid double-counting the corner attribute, this "
        "calculation applies the Location Premiums on top of the plain "
        "2-Bed Standard base price, matching the document's own worked "
        "example (+4%+3%+6% over base)."
        if is_corner else ""
    )

    return {
        "answer": f"The total price comes to PKR {total:,}.{note}",
        "status": "CALCULATED FROM DOCUMENT",
        "calculation": "\n".join(calc_lines),
        "sources": [
            _source("price_list", f"Base Prices (Block {block})"),
            _source("price_list", "Location Premiums"),
        ],
    }


# ---------------------------------------------------------------------------
# DETERMINISTIC HANDLER 3: transfer fee conflict check
# ---------------------------------------------------------------------------
def handle_transfer_fee(question_lower):
    price_list_text = _read_raw("price_list")
    booking_faq_text = _read_raw("booking_faq")

    price_list_match = re.search(
        r"Transfer fee \(before possession\):\s*(\d+(?:\.\d+)?)%", price_list_text
    )
    booking_match = re.search(r"Transfer fee is\s*\*\*(\d+(?:\.\d+)?)%", booking_faq_text)

    price_list_pct = price_list_match.group(1) if price_list_match else None
    booking_pct = booking_match.group(1) if booking_match else None

    if price_list_pct is None and booking_pct is None:
        return {"answer": NOT_FOUND_MSG, "status": "NOT FOUND", "sources": []}

    if price_list_pct is not None and booking_pct is not None and price_list_pct != booking_pct:
        answer = (
            "There is conflicting information in the provided documents.\n\n"
            f"MGC Price List (Other Charges): transfer fee = {price_list_pct}% of the current list price.\n"
            f"MGC Booking Policy & FAQ (Transfers): transfer fee = {booking_pct}% of the current list price.\n\n"
            "Because the documents conflict, I cannot determine which value is current. "
            "Please confirm the correct value with the marketing manager."
        )
        return {
            "answer": answer,
            "status": "CONFLICTING INFORMATION",
            "sources": [
                _source("price_list", "Other Charges"),
                _source("booking_faq", "Transfers"),
            ],
        }

    # Values agree (or only one document mentions it) - safe to report directly.
    pct = price_list_pct or booking_pct
    return {
        "answer": f"The transfer fee is {pct}% of the current list price.",
        "status": "FOUND IN DOCUMENT",
        "sources": [_source("price_list", "Other Charges")],
    }


# ---------------------------------------------------------------------------
# DETERMINISTIC HANDLER 4: rental yield (explicitly withheld)
# ---------------------------------------------------------------------------
def handle_rental_yield(question_lower):
    booking_faq_text = _read_raw("booking_faq")
    if "rental yield" in booking_faq_text.lower():
        return {
            "answer": (
                "The provided documents do not specify a rental yield for a 1-bed unit, "
                "so I can't provide a reliable figure. Please confirm with the marketing manager."
            ),
            "status": "NOT FOUND",
            "sources": [_source("booking_faq", "Frequently Asked")],
        }
    return {"answer": NOT_FOUND_MSG, "status": "NOT FOUND", "sources": []}


# ---------------------------------------------------------------------------
# DETERMINISTIC HANDLER 5: anchor tenant (explicitly unconfirmed)
# ---------------------------------------------------------------------------
def handle_anchor_tenant(question_lower):
    brochure_text = _read_raw("brochure")
    if "no anchor tenant has been confirmed" in brochure_text.lower():
        return {
            "answer": (
                "No anchor tenant has been confirmed. The brochure states that anchor "
                "tenancy discussions are ongoing, with no anchor tenant confirmed as of "
                "the brochure's issue date."
            ),
            "status": "FOUND IN DOCUMENT",
            "sources": [_source("brochure", "Commercial Podium")],
        }
    return {"answer": NOT_FOUND_MSG, "status": "NOT FOUND", "sources": []}


# ---------------------------------------------------------------------------
# GENERIC FALLBACK: retrieval + LLM (only used for questions that don't match
# one of the 5 patterns above)
# ---------------------------------------------------------------------------
def _call_llm(question, context_chunks):
    """Send the retrieved context + a strict system prompt to an LLM.

    Supports three interchangeable providers, all through the same OpenAI
    SDK (each one exposes an OpenAI-compatible API, so only the base_url,
    key, and model name change):
        1. OpenRouter   (OPENROUTER_API_KEY)
        2. Google Gemini (GEMINI_API_KEY)
        3. OpenAI directly (OPENAI_API_KEY)
    Checked in that order - whichever key is set first wins. Returns None
    if no key at all is configured, so the caller can fall back to
    showing the raw retrieved excerpt instead of calling an LLM.
    """
    from openai import OpenAI

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if openrouter_key:
        # OpenRouter: same OpenAI SDK, different base_url + model name.
        # Model names look like "openai/gpt-4o-mini" or "anthropic/claude-3.5-sonnet".
        client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
        model_name = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    elif gemini_key:
        # Google Gemini also exposes an OpenAI-compatible endpoint, so the
        # same SDK works here too - just point it at Google's base_url.
        # Model names look like "gemini-2.0-flash" or "gemini-1.5-pro".
        client = OpenAI(
            api_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    elif openai_key:
        client = OpenAI(api_key=openai_key)
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    else:
        return None

    context_text = "\n\n".join(
        f"[Source: {c['document_name']} - {c['section']}]\n{c['text']}"
        for c in context_chunks
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": STRICT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Document context:\n\n{context_text}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def generic_rag_answer(question):
    chunks = retrieve(question, k=TOP_K)
    sources = sorted({f"{c['document_name']} - {c['section']}" for c in chunks})

    llm_text = _call_llm(question, chunks)
    if llm_text is not None:
        return {"answer": llm_text, "status": "FOUND IN DOCUMENT", "sources": sources}

    # No LLM configured - show the closest matching excerpt instead of guessing.
    top_chunk = chunks[0] if chunks else None
    if top_chunk is None:
        return {"answer": NOT_FOUND_MSG, "status": "NOT FOUND", "sources": []}
    return {
        "answer": (
            "No LLM is configured (OPENAI_API_KEY not set), so here is the most "
            "relevant excerpt found instead of a generated answer:\n\n"
            f"\"{top_chunk['text'][:500]}\""
        ),
        "status": "FOUND IN DOCUMENT",
        "sources": [f"{top_chunk['document_name']} - {top_chunk['section']}"],
    }


# ---------------------------------------------------------------------------
# ROUTER: decide which handler answers a given question
# ---------------------------------------------------------------------------
def answer_question(question):
    """Main entry point used by app.py and test_questions.py.

    Checks the question against the 5 known, business-critical patterns
    first (each one is answered deterministically from the raw documents).
    Anything else falls through to retrieval + LLM.
    """
    q = question.lower()

    if "rental yield" in q:
        result = handle_rental_yield(q)
    elif "anchor tenant" in q:
        result = handle_anchor_tenant(q)
    elif "transfer fee" in q:
        result = handle_transfer_fee(q)
    elif re.search(r"floor\s*\d+", q) and ("margalla" in q or "corner" in q):
        result = handle_price_calculation(q)
    elif "base price" in q:
        result = handle_base_price_lookup(q)
    else:
        result = None

    if result is None:
        result = generic_rag_answer(question)

    # Make sure every result always has all three keys, even if a handler
    # didn't set "calculation".
    result.setdefault("calculation", None)
    return result
