import PyPDF2
import docx
import os
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.secret_key = "super_secret_key"  # Nodig voor sessiebeheer

vectorstore = None
SUPERUSER_CREDENTIALS = {"username": "admin", "password": "securepassword"}  # Superuser login

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

def process_uploaded_file(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == SUPERUSER_CREDENTIALS["username"] and password == SUPERUSER_CREDENTIALS["password"]:
            session["superuser"] = True
            return redirect(url_for("upload_page"))
        else:
            return "Ongeldige inloggegevens", 403
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("superuser", None)
    return redirect(url_for("index"))

@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    if "superuser" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        global vectorstore
        text = process_uploaded_file(file_path)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts([text], embeddings)
        
        return jsonify({"message": "File uploaded and processed successfully"})
    return render_template("upload.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    global vectorstore
    if vectorstore is None:
        return jsonify({"error": "No documents uploaded yet"}), 400
    
    data = request.get_json()
    question = data.get("question", "")
    
    llm = OpenAI()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(), return_source_documents=True)
    
    result = qa_chain({"query": question})
    response = result["result"]
    sources = "\n".join([doc.metadata.get("source", "Onbekende bron") for doc in result["source_documents"]])
    
    return jsonify({"answer": response, "sources": sources})

if __name__ == "__main__":
    app.run(debug=True)
