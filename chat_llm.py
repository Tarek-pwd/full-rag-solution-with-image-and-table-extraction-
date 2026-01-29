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
   - Even if the content consists ONLY of images or ONLY of tables.
   - "text_out" must NEVER be empty, null, or an empty string.

2. "text_out" must be written in clear, natural language and MUST:
   - Explain what the document content represents.
   - Explicitly describe images and tables.
   - Use phrasing such as:
     • "The following image shows..."
     • "The image illustrates..."
     • "The table below presents..."
     • "This document discusses..."

3. Images handling:
   - ALWAYS describe what each image depicts in "text_out".
   - Include image_path in the "images" field.
   - Do NOT copy image descriptions verbatim; summarize intelligently.

4. Tables handling:
   - Explain in "text_out" what each table is about.
   - Do NOT perform numerical summaries unless explicitly requested.
   - Return tables EXACTLY as provided in valid HTML.

5. Output format:
   - Return a valid JSON object with EXACTLY these keys:
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

9. Coverage requirement:
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
        # Include above and below text if available
        above = chunk.get("text_above", "").strip()
        below = chunk.get("text_below", "").strip()
        return (
            f"[IMAGE]\n"
            f"Page: {chunk.get('page number')}\n"
            f"Path: {chunk.get('image_path')}\n"
            f"Description:\n{chunk.get('description')}\n"
            f"Text Above:\n{above}\n"
            f"Text Below:\n{below}"
        )

    # -------- TABLE --------
    if chunk_type == "table":
        above = chunk.get("text_above", "").strip()
        below = chunk.get("text_below", "").strip()
        return (
            f"[TABLE]\n"
            f"Page: {chunk.get('page number')}\n"
            f"Title: {chunk.get('description')}\n"
            f"Table Content:\n{chunk.get('table content')}\n"
            f"Text Above:\n{above}\n"
            f"Text Below:\n{below}"
        )

    return "[UNKNOWN CHUNK FORMAT]"

# =========================
# MESSAGE STATE
# =========================
messages = [{"role": "system", "content": SYS_PROMPT}]

# =========================
# CHAT FUNCTION
# =========================
def chat(user_query: str):
    retrieved_chunks = RAG_query(user_query)

    # ---- FORCE IMAGE DESCRIPTIONS ALWAYS ----
    filtered_chunks = []
    for chunk in retrieved_chunks:
        if chunk.get("type") == "image":
            filtered_chunks.append(chunk)  # always include
        elif chunk.get("type") == "table":
            filtered_chunks.append(chunk)
        elif "text" in chunk:
            filtered_chunks.append(chunk)

    # Serialize all chunks for the model
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
