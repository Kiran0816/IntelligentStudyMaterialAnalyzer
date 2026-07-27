import os
# Set environment variable to prevent libiomp5md.dll OpenMP conflicts on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, render_template, request, jsonify, send_file
import io
import logging
import threading
from werkzeug.utils import secure_filename

# Import database and AI modules
from modules.database import (
    init_db, save_upload, update_upload_texts, get_upload, 
    get_all_uploads, delete_upload, save_summary, get_summary, 
    save_mcqs, get_mcqs, save_qa, get_qa_history, save_keywords, 
    get_keywords, save_analytics, get_analytics
)
from modules.ocr_module import extract_text_from_file
from modules.difficulty_analyzer import analyze_difficulty
from modules.keyword_extractor import extract_keywords
from modules.summarizer import generate_summary
from modules.qa_system import answer_question
from modules.mcq_generator import generate_mcqs

# Import feature blueprints and setup helpers
from agent_routes import agent_bp, init_agent      # Feature 1: Agent
from url_routes import url_bp, init_url_feature    # Feature 2: URL Ingestion
from voice_routes import voice_bp                  # Feature 3: Voice Assistant

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB limit for audio uploads

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Database on startup
init_db()
init_url_feature()                  # Feature 2: creates url_metadata table
app.register_blueprint(url_bp)       # Feature 2: registers /api/url/* routes
init_agent(app)                     # Feature 1: sets up LangGraph agent
app.register_blueprint(agent_bp)     # Feature 1: registers /api/agent/* routes
app.register_blueprint(voice_bp)     # Feature 3: registers /api/voice/* routes

# Pre-download Whisper model asynchronously at startup
def _preload_whisper():
    try:
        from modules.stt_service import _get_model
        _get_model()
    except Exception as e:
        logger.warning(f"Failed to preload Whisper model: {e}")
threading.Thread(target=_preload_whisper, daemon=True).start()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/uploads', methods=['GET'])
def list_uploads():
    """Returns a list of all uploaded study materials."""
    try:
        uploads = get_all_uploads()
        return jsonify({"success": True, "uploads": uploads})
    except Exception as e:
        logger.error(f"Error listing uploads: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Uploads a file and extracts raw text."""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part in the request"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        file_ext = os.path.splitext(filename)[1].lower()
        file_type = "PDF" if file_ext == ".pdf" else "Image"
        
        try:
            # 1. Save record in database (pending text extraction)
            upload_id = save_upload(filename, filepath, file_type)
            
            # 2. Extract text (PDF/Image OCR or plain text extraction)
            logger.info(f"Extracting text from uploaded file: {filename}")
            raw_text = extract_text_from_file(filepath)
            
            if not raw_text or not raw_text.strip():
                # Delete the physical file and db record since it's unreadable
                delete_upload(upload_id)
                return jsonify({"success": False, "error": "Could not extract any readable text from this file."}), 422
                
            # 3. Clean raw text for processed text
            processed_text = raw_text.strip()
            
            # 4. Save extracted text back to database
            update_upload_texts(upload_id, raw_text, processed_text)
            
            return jsonify({
                "success": True, 
                "upload_id": upload_id, 
                "filename": filename,
                "message": "File uploaded and text extracted successfully."
            })
            
        except Exception as e:
            logger.error(f"Error during file processing: {e}")
            # Clean up uploaded file if it failed
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass
            return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/analyze/<int:upload_id>', methods=['POST'])
def analyze_file(upload_id):
    """
    Performs full analysis (Keywords, Difficulty, Summarization, MCQs).
    Returns cached results if already analyzed.
    """
    try:
        upload = get_upload(upload_id)
        if not upload:
            return jsonify({"success": False, "error": "File not found"}), 404
            
        processed_text = upload['processed_text']
        if not processed_text:
            return jsonify({"success": False, "error": "No text content available to analyze"}), 400
            
        # Check if already analyzed (by checking if analytics table has data)
        analytics = get_analytics(upload_id)
        summary = get_summary(upload_id)
        mcqs = get_mcqs(upload_id)
        keywords = get_keywords(upload_id)
        
        if analytics and summary and mcqs and keywords:
            logger.info(f"Returning cached analysis results for upload_id={upload_id}")
            return jsonify({
                "success": True,
                "data": {
                    "analytics": analytics,
                    "summary": summary,
                    "mcqs": mcqs,
                    "keywords": [kw[0] for kw in keywords] # return just list of words
                }
            })
            
        logger.info(f"Running fresh analysis for upload_id={upload_id}")
        
        # 1. Keywords Extraction
        kw_list = extract_keywords(processed_text, top_n=10)
        save_keywords(upload_id, kw_list)
        
        # 2. Difficulty & Study Time Analysis
        diff_data = analyze_difficulty(processed_text)
        save_analytics(
            upload_id, 
            diff_data['difficulty_level'], 
            diff_data['sentence_count'], 
            diff_data['word_count'], 
            diff_data['estimated_study_time']
        )
        
        # 3. Summarization
        summary_text = generate_summary(processed_text)
        save_summary(upload_id, summary_text)
        
        # 4. MCQ Generation (Generate up to 5 MCQs)
        mcq_list = generate_mcqs(processed_text, count=5)
        save_mcqs(upload_id, mcq_list)
        
        return jsonify({
            "success": True,
            "data": {
                "analytics": diff_data,
                "summary": summary_text,
                "mcqs": mcq_list,
                "keywords": [kw[0] for kw in kw_list]
            }
        })
        
    except Exception as e:
        logger.error(f"Error during analysis of upload_id={upload_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/ask/<int:upload_id>', methods=['POST'])
def ask_question_route(upload_id):
    """Asks a question about the study document."""
    try:
        data = request.json
        if not data or 'question' not in data:
            return jsonify({"success": False, "error": "No question provided"}), 400
            
        question = data['question'].strip()
        if not question:
            return jsonify({"success": False, "error": "Question cannot be empty"}), 400
            
        upload = get_upload(upload_id)
        if not upload:
            return jsonify({"success": False, "error": "File not found"}), 404
            
        processed_text = upload['processed_text']
        if not processed_text:
            return jsonify({"success": False, "error": "No text content available to query"}), 400
            
        # Run Q&A engine
        answer = answer_question(processed_text, question)
        
        # Save to Q&A history
        save_qa(upload_id, question, answer)
        
        return jsonify({
            "success": True,
            "question": question,
            "answer": answer
        })
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/qa-history/<int:upload_id>', methods=['GET'])
def get_qa_history_route(upload_id):
    """Gets Q&A history for a specific document."""
    try:
        history = get_qa_history(upload_id)
        return jsonify({"success": True, "history": history})
    except Exception as e:
        logger.error(f"Error fetching QA history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/delete/<int:upload_id>', methods=['POST'])
def delete_file_route(upload_id):
    """Deletes an uploaded file and all its records."""
    try:
        delete_upload(upload_id)
        return jsonify({"success": True, "message": "File and analysis results deleted."})
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/download/<int:upload_id>', methods=['GET'])
def download_results(upload_id):
    """Downloads a formatted Markdown study guide of the analyzed material."""
    try:
        upload = get_upload(upload_id)
        if not upload:
            return jsonify({"success": False, "error": "File not found"}), 404
            
        analytics = get_analytics(upload_id)
        summary = get_summary(upload_id)
        mcqs = get_mcqs(upload_id)
        keywords = get_keywords(upload_id)
        
        if not analytics or not summary:
            return jsonify({"success": False, "error": "File has not been fully analyzed yet."}), 400
            
        # Build Markdown document content
        md = []
        md.append(f"# Study Analyzer Report: {upload['filename']}")
        md.append(f"*File Type:* {upload['file_type']} | *Analyzed On:* {analytics.get('created_at', 'N/A')}\n")
        md.append("## Reading Analytics")
        md.append(f"- **Difficulty Rating:** {analytics['difficulty_level']}")
        md.append(f"- **Word Count:** {analytics['word_count']} words")
        md.append(f"- **Sentence Count:** {analytics['sentence_count']} sentences")
        md.append(f"- **Estimated Study Time:** {analytics['estimated_study_time']} minutes")
        md.append("")
        
        if keywords:
            md.append("## Key Study Concepts")
            kws = ", ".join([f"`{kw[0]}`" for kw in keywords])
            md.append(kws)
            md.append("")
            
        md.append("## Document Summary")
        md.append(summary)
        md.append("")
        
        if mcqs:
            md.append("## Practice Multiple-Choice Questions (MCQs)")
            for i, mcq in enumerate(mcqs, 1):
                md.append(f"### Q{i}. {mcq['question']}")
                for opt in mcq['options']:
                    md.append(f"- [ ] {opt}")
                md.append(f"\n*Correct Answer:* **{mcq['correct_answer']}**")
                md.append("")
                
        # Send as an in-memory file attachment
        filename_base = os.path.splitext(upload['filename'])[0]
        output_filename = f"{filename_base}_study_guide.md"
        
        mem_file = io.BytesIO()
        mem_file.write(("\n".join(md)).encode('utf-8'))
        mem_file.seek(0)
        
        return send_file(
            mem_file,
            as_attachment=True,
            download_name=output_filename,
            mimetype="text/markdown"
        )
    except Exception as e:
        logger.error(f"Error downloading results: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
