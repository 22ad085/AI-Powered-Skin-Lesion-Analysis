import os
import numpy as np
import tensorflow as tf
import sqlite3
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file, session, send_from_directory
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
from reportlab.platypus import Image as RLImage, Table, TableStyle
from reportlab.lib import colors

app = Flask(__name__)
app.secret_key = "secret_key_for_session"

# Folders
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["REPORT_FOLDER"] = "reports"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["REPORT_FOLDER"], exist_ok=True)

# Load model
try:
    model = tf.keras.models.load_model("skin_cancer_model.h5")
    print("✅ Model loaded successfully.")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Database initialization
def init_db():
    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  filename TEXT,
                  label TEXT,
                  confidence REAL,
                  risk TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    username = request.form.get("username", "Anonymous")
    session["username"] = username

    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    session["uploaded_file"] = filename
    return redirect(url_for("predict"))

@app.route("/predict")
def predict():
    filename = session.get("uploaded_file", None)
    if not filename:
        return redirect(url_for("index"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    if model is None:
        return "Model not loaded. Please check the model file.", 500

    # Preprocess image
    img = tf.keras.utils.load_img(filepath, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    confidence = float(prediction)

    # Determine label and risk
    if confidence > 0.5:
        label = "Malignant"
        risk = "High Risk"
    else:
        label = "Benign"
        risk = "Low Risk"

    # Save prediction for PDF
    session["last_label"] = label
    session["last_confidence"] = confidence
    session["last_risk"] = risk

    # Store in database
    username = session.get("username", "Anonymous")
    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    c.execute('''
        INSERT INTO results (username, filename, label, confidence, risk)
        VALUES (?, ?, ?, ?, ?)
    ''', (username, filename, label, confidence, risk))
    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        filename=filename,
        uploaded_image_path=url_for('uploaded_file', filename=filename),
        result=label,
        confidence=confidence,
        risk=risk
    )

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/download/<filename>")
def download_report(filename):
    report_path = os.path.join(app.config["REPORT_FOLDER"], f"Melanoma_Report_{filename}.pdf")

    # 🔹 Fetch details from database
    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    c.execute("SELECT username, label, confidence, risk, timestamp FROM results WHERE filename = ? ORDER BY id DESC LIMIT 1", (filename,))
    row = c.fetchone()
    conn.close()

    if not row:
        return "No record found for this file.", 404

    username, label, confidence, risk, timestamp = row

    # 🔹 Create PDF document
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # --- Title ---
    story.append(Paragraph("Melanoma Analysis Report", styles["Title"]))
    story.append(Spacer(1, 20))

    # --- Add Patient Info Table ---
    data = [
        ["Patient Name:", username],
        ["Image Name:", filename],
        ["Prediction:", label],
        ["Confidence:", f"{round(confidence * 100, 2)}%"],
        ["Risk Level:", risk],
        ["Tested On:", timestamp],
    ]

    table = Table(data, colWidths=[120, 350])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # --- Add Uploaded Image ---
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(img_path):
        story.append(Paragraph("Uploaded Lesion Image:", styles["Heading3"]))
        story.append(Spacer(1, 12))
        story.append(RLImage(img_path, width=250, height=250))  # resize image
        story.append(Spacer(1, 20))


    # Build PDF
    doc.build(story)

    return send_file(report_path, as_attachment=True)

@app.route("/download_database")
def download_database():
    conn = sqlite3.connect("results.db")
    df = pd.read_sql_query("SELECT * FROM results", conn)
    conn.close()
    output = "results.xlsx"
    df.to_excel(output, index=False)
    return send_file(output, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
