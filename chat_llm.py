import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from query_doc import RAG_query

# =========================
# ENV + CLIENT
# =========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# SYSTEM PROMPT
# =========================
SYS_PROMPT = """
You are a document-grounded assistant.

You are provided with content extracted from documents. The content may include:
- Text passages
- Images (with image_path and image_description)
- Tables (in HTML format)

========================
CRITICAL RULES (MANDATORY)
========================

1. You MUST ALWAYS produce a NON-EMPTY explanatory response in the field "text_out".
   - This applies even if the content consists ONLY of images or ONLY of tables.
   - "text_out" must NEVER be empty, null, or an empty string.

2. "text_out" must be written in clear, natural language and MUST:
   - Explain what the document content represents.
   - Explicitly refer to images and tables when they exist.
   - Use phrasing such as:
     • "The following image shows..."
     • "The image illustrates..."
     • "The table below presents..."
     • "This document discusses..."

3. Images handling:
   - If images are present, describe what each image depicts in "text_out".
   - Include image_path ONLY inside the "images" field.

4. Tables handling:
   - If tables are present, explain in "text_out" what each table is about.
   - Do NOT perform numerical summaries unless explicitly requested.
   - Return tables EXACTLY as provided in valid HTML.

5. Output format:
   - You MUST return a valid JSON object with EXACTLY these keys:
     {
       "text_out": string (NON-EMPTY),
       "images":   list of { "image_path", "image_description" },
       "tables":   list of HTML strings
     }
   - If a field has no content, return an empty list [] — NEVER null.

6. Authority & tone:
   - NEVER mention retrieval mechanisms, RAG, embeddings, FAISS, or chunks.
   - Speak with confidence and authority.

7. Content priority:
   - Earlier content is more important and must be emphasized first.

8. Insufficient content:
   - If content is minimal or irrelevant, still produce a helpful response.

10. Coverage requirement:
    - Combine ALL relevant text into one coherent explanation.

========================
END OF RULES
========================
"""

# =========================
# CHUNK SERIALIZATION
# =========================
def serialize_chunk(chunk: dict) -> str:
    chunk_type = chunk.get("type", "text")

    # -------- TEXT --------
    if chunk_type == "text" or "text" in chunk:
        header = chunk.get("header", "").strip()
        texts = chunk.get("text", [])
        pages = chunk.get("pages", [])

        lines = [f"[HEADER] {header}"]
        for t, p in zip(texts, pages):
            lines.append(f"(Page {p}) {t.strip()}")
        return "\n".join(lines)

    # -------- IMAGE --------
    if chunk_type == "image":
        return (
            f"[IMAGE]\n"
            f"Page: {chunk.get('page number')}\n"
            f"Path: {chunk.get('image_path')}\n"
            f"Description:\n{chunk.get('description')}"
        )

    # -------- TABLE --------
    if chunk_type == "table":
        return (
            f"[TABLE]\n"
            f"Page: {chunk.get('page number')}\n"
            f"Title: {chunk.get('description')}\n"
            f"HTML:\n{chunk.get('table content')}"
        )

    return "[UNKNOWN CHUNK FORMAT]"

# =========================
# MESSAGE STATE
# =========================
messages = [
    {"role": "system", "content": SYS_PROMPT}
]

# =========================
# CHAT FUNCTION
# =========================
def chat(user_query: str):
    """
    user_query: the question or instruction for the assistant
    returns: JSON object with keys: text_out, images, tables
    """

    # ---- detect if user wants images ----
    wants_images = any(
        kw in user_query.lower()
        for kw in ["image", "images", "figure", "photo", "picture", "show"]
    )

    # ---- retrieve relevant document chunks ----
    retrieved_chunks = RAG_query(user_query)

    # ---- FILTER IMAGES AT SOURCE ----
    filtered_chunks = []
    for chunk in retrieved_chunks:
        if chunk.get("type") == "image" and not wants_images:
            continue
        filtered_chunks.append(chunk)

    # ---- combine chunks for input to the model ----
    rag_block = "\n\n---\n\n".join(
        serialize_chunk(chunk) for chunk in filtered_chunks
    )

    user_message = f"""
USER QUESTION:
{user_query}

DOCUMENT CONTENT:
{rag_block}
"""

    messages.append({"role": "user", "content": user_message})

    # ---- call OpenAI API ----
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"}  # 🔒 HARD JSON GUARANTEE
    )

    assistant_reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_reply})

    print("\n=== MODEL OUTPUT ===")
    print(assistant_reply)

    # ---- HARD VALIDATION ----
    try:
        return json.loads(assistant_reply)
    except json.JSONDecodeError:
        raise RuntimeError(
            "❌ Model violated JSON contract.\n\nRaw output:\n" + assistant_reply
        )


# =========================
# TEST CALL (OPTIONAL)
# =========================
# result = chat("Summarize the document")
# print(result)
