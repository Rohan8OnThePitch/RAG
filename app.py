from flask import Flask, request, jsonify, render_template
import os
from documents import process_docx, process_pdf, process_txt
from indexing import index_document
from querying import query_documents
import preprocess

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route to render index.html (upload page)
@app.route('/')
def index():
    return render_template('index.html')  # Serves the HTML form

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
        
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # Process file into structured chunks
    if file.filename.endswith('.docx'):
        chunks = process_docx(file_path)
    elif file.filename.endswith('.pdf'):
        chunks = process_pdf(file_path)
    elif file.filename.endswith('.txt'):
        chunks = process_txt(file_path)
    else:
        return jsonify({"error": "Unsupported file type"}), 400
        
    # Now chunks contain section and subsection information
    # We don't need to join them into full_text anymore
    result = index_document("documents", file.filename, chunks)
    
    return jsonify(result)

@app.route('/query', methods=['POST'])
def query():
    try:
        print("Entering query route")
        query_text = request.json.get('query')
        
        if not query_text:
            return jsonify({"error": "Query cannot be empty"}), 400
            
        score_threshold = request.json.get('score_threshold', 0.5)
        results = query_documents("documents", query_text, score_threshold=score_threshold)
       # print(results)
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in query route: {str(e)}")  # Debug log
        return jsonify({"error": f"Query processing failed: {str(e)}"}), 500
if __name__ == '__main__':
    app.run(debug=True, port=5000)