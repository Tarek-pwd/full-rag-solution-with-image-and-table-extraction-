import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

FAISS_INDEX_PATH = "document_faiss.index"
META_PATH = "document_chunks.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_FAISS_index(chunks_array):
    embedding_texts = []
    metadata = []

    for chunk in chunks_array:

        # ============================================================
        # TEXT CHUNKS (merged paragraphs under headers)
        # ============================================================
        if "text" in chunk and chunk["text"]:
            all_text = chunk["text"]
            all_pages = chunk["pages"]
            header = chunk.get("header", "").strip()

            prev_page = all_pages[0]
            text_acc = ""
            full_doc = ""

            for i in range(len(all_text)):
                page = all_pages[i]
                sentence = all_text[i]

                if page == prev_page:
                    text_acc += " " + sentence
                else:
                    full_doc += f"Page: {prev_page}\nHeader: {header}\n{text_acc.strip()}\n\n"
                    text_acc = sentence

                prev_page = page

            # flush last
            full_doc += f"Page: {prev_page}\nHeader: {header}\n{text_acc.strip()}"

            embedding_texts.append(full_doc)
            metadata.append(chunk)

            print("✅ TEXT CHUNK INDEXED\n", full_doc[:200], "\n")

        # ============================================================
        # IMAGE CHUNKS (description + above + below)
        # ============================================================
        elif chunk.get("type") == "image":
            page = chunk.get("page number", "")
            desc = chunk.get("description", "").strip()
            above = chunk.get("text_above", "").strip()
            below = chunk.get("text_below", "").strip()

            combined = f"""
            Page: {page}
            [IMAGE]
            Description:
            {desc}

            Text Above:
            {above}

            Text Below:
            {below}
            """.strip()

            if combined:
                embedding_texts.append(combined)
                metadata.append(chunk)

                print("🖼 IMAGE CHUNK INDEXED\n", combined[:200], "\n")

        # ============================================================
        # TABLE CHUNKS (caption + content + above + below)
        # ============================================================
        elif chunk.get("type") == "table":
            page = chunk.get("page number", "")
            desc = chunk.get("description", "").strip()
            table_content = chunk.get("table content", "")
            above = chunk.get("text_above", "").strip()
            below = chunk.get("text_below", "").strip()

            combined = f"""
            Page: {page}
            [TABLE]
            Caption:
            {desc}

            Table Content:
            {table_content}

            Text Above:
            {above}

            Text Below:
            {below}
            """.strip()

            if combined:
                embedding_texts.append(combined)
                metadata.append(chunk)

                print("📊 TABLE CHUNK INDEXED\n", combined[:200], "\n")

    # ============================================================
    # FAISS BUILD
    # ============================================================
    print(f"[INFO] Total indexed chunks: {len(embedding_texts)}")

    embeddings = model.encode(
        embedding_texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("✅ Saved document_faiss.index")
    print("✅ Saved document_chunks.pkl")
