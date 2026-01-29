from flask import Flask, render_template, request, jsonify,  send_from_directory
import os
import threading
import time
from ingest_doc import ingest_document
from chat_llm import chat

app = Flask(__name__)

UPLOAD_FOLDER = "uploads/documents"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------
# GLOBAL PROGRESS STATE
# ----------------------------
ingestion_status = {
    "running": False,
    "progress": 0,
    "done": False
}

# ----------------------------
# BACKGROUND INGESTION
# ----------------------------
def run_ingestion(pdf_path):
    ingestion_status["running"] = True
    ingestion_status["progress"] = 0
    ingestion_status["done"] = False

    def update_progress(p):
        ingestion_status["progress"] = p

    ingest_document(
        pdf_path,
        progress_callback=update_progress
    )

    ingestion_status["progress"] = 100
    ingestion_status["running"] = False
    ingestion_status["done"] = True



# ----------------------------
# ROUTES
# ----------------------------
@app.route("/")
def upload_page():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_pdf():
    file = request.files["pdf"]
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    thread = threading.Thread(
        target=run_ingestion,
        args=(save_path,)
    )
    thread.start()

    return jsonify({"status": "started"})


@app.route("/progress")
def progress():
    return jsonify(ingestion_status)

@app.route('/generated/images/<path:filename>')
def serve_generated_images(filename):
    return send_from_directory('generated/images', filename)

@app.route("/chat")
def chat_page():
    return render_template("chat.html")


@app.route('/ask', methods=['POST'])
def chat_endpoint():
    data = request.get_json()
    result = chat(data['message'])
    
    # Transform images for frontend
    images = []
    for img in result.get("images", []):
        image_path = img.get("image_path", "")
        # Just prepend the route path
        images.append({
            "src": f"/generated/images/{image_path}",
            "caption": img.get("image_description", "")
        })
    
    return jsonify({
        "response": result.get("text_out", ""),
        "images": images,
        "tables": result.get("tables", [])
    })

if __name__ == "__main__":
    app.run(debug=True)
