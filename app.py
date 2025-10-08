from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from werkzeug.utils import secure_filename
import shutil
from Analyzer import analyze_and_generate_report

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
CHART_PATH = os.path.join('static', 'result_chart.png')
REPORT_PATH = os.path.join('static', 'result.docx')  # For download

# Ensure required folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

# ---------------------
# Route: Home Page
# ---------------------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# ---------------------
# Route: Compare & Analyze
# ---------------------
@app.route('/compare', methods=['POST'])
def compare():
    # Clear previous uploads
    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
    os.makedirs(UPLOAD_FOLDER)

    for key in request.files:
        files = request.files.getlist(key)
    
    # Extract actual year from form field names like "files_2018" or "images_2020"
        digits = ''.join(filter(str.isdigit, key))
        year = digits if len(digits) == 4 else None

        if not year:
            print(f"Skipping invalid upload group: {key}")
            continue

        year_folder = os.path.join(UPLOAD_FOLDER, year)
        os.makedirs(year_folder, exist_ok=True)

        for file in files:
            if file:
                filename = secure_filename(file.filename)
                file.save(os.path.join(year_folder, filename))

    # Analyze & Generate Chart + Report
    analyze_and_generate_report(UPLOAD_FOLDER, RESULT_FOLDER, CHART_PATH)

    # Move final report to static for download
    shutil.copy(os.path.join(RESULT_FOLDER, "final_report.docx"), REPORT_PATH)

    return render_template("result.html", chart_url=CHART_PATH)

# ---------------------
# Route: Download Report
# ---------------------
@app.route('/download')
def download():
    return send_file(REPORT_PATH, as_attachment=True)

import webbrowser
import threading

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

threading.Timer(1.25, open_browser).start()

# ---------------------
# Launch Flask App
# ---------------------
if __name__ == '__main__':
    app.run(debug=True)
