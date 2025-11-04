import os
import sqlite3
import cv2
import pygame
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from gtts import gTTS
import google.generativeai as genai
from googletrans import Translator
import tempfile
import base64
from PIL import Image, ImageDraw
import fitz  # PyMuPDF for PDF processing
import io
import mediapipe as mp
import pyautogui
import threading
import time
import json

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['AUDIO_FOLDER'] = 'audio'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

# Configure Gemini AI
genai.configure(api_key='AIzaSyCDtMEOvDtjjAN4d9QfPv9fXSxG6xtP7_o')
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# Initialize translator
translator = Translator()

# Initialize pygame for audio
pygame.mixer.init()

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'txt', 'doc', 'docx'}

# Global variables for eye tracking
eye_tracking_active = False
gaze_coordinates = (0, 0)
selected_text = ""
tracking_thread = None
current_document_data = None
document_regions = []

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def process_image_with_genai(filepath):
    try:
        sample_file = genai.upload_file(path=filepath)
        text = "Extract all text from this image accurately. Return only the text without any additional commentary. Preserve the layout and structure of the text."
        response = gemini_model.generate_content([text, sample_file])
        return response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"

def extract_text_from_region(image_path, region):
    """Extract text from a specific region of the image"""
    try:
        # Load the image
        image = Image.open(image_path)
        
        # Convert region coordinates (screen coordinates to image coordinates)
        screen_width, screen_height = pyautogui.size()
        img_width, img_height = image.size
        
        # Calculate scaling factors
        scale_x = img_width / screen_width
        scale_y = img_height / screen_height
        
        # Define region around gaze point (adjustable size)
        region_size = 300  # pixels in screen coordinates
        region_x = max(0, int(region[0] * scale_x - region_size/2))
        region_y = max(0, int(region[1] * scale_y - region_size/2))
        region_width = min(region_size, img_width - region_x)
        region_height = min(region_size, img_height - region_y)
        
        # Crop the region
        cropped_image = image.crop((region_x, region_y, region_x + region_width, region_y + region_height))
        
        # Save cropped image temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_region.png')
        cropped_image.save(temp_path)
        
        # Extract text from cropped region using Gemini
        region_text = process_image_with_genai(temp_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return region_text.strip()
        
    except Exception as e:
        print(f"Region text extraction error: {e}")
        return ""

def pdf_to_image(pdf_path):
    """Convert PDF to image for display"""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)  # First page
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        doc.close()
        return img_data
    except Exception as e:
        print(f"PDF to image error: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with position information"""
    try:
        doc = fitz.open(pdf_path)
        text_data = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_instances = page.get_text("dict")
            
            for block in text_instances["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_data.append({
                                'text': span['text'],
                                'bbox': span['bbox'],  # [x0, y0, x1, y1]
                                'page': page_num
                            })
        
        doc.close()
        return text_data
    except Exception as e:
        print(f"PDF text extraction error: {e}")
        return []

def find_text_at_position(text_data, x, y, page_num=0):
    """Find text at specific coordinates in PDF"""
    try:
        for text_item in text_data:
            if text_item['page'] == page_num:
                bbox = text_item['bbox']
                # Check if coordinates are within this text block
                if (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):
                    return text_item['text']
        return ""
    except Exception as e:
        print(f"Text position search error: {e}")
        return ""

def text_to_speech(text, lang='en', filename='output.mp3'):
    """Convert text to speech using gTTS"""
    try:
        # Limit text length for TTS
        if len(text) > 500:
            text = text[:500] + "..."
            
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_path = os.path.join(app.config['AUDIO_FOLDER'], filename)
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def play_audio(audio_path):
    """Play audio using pygame"""
    try:
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
    except Exception as e:
        print(f"Audio play error: {e}")

def start_eye_tracking():
    """Start eye tracking in a separate thread"""
    global eye_tracking_active, gaze_coordinates
    
    cam = cv2.VideoCapture(0)
    screen_w, screen_h = pyautogui.size()
    
    # Variables for gaze stabilization
    gaze_history = []
    max_history = 10
    click_cooldown = 0
    last_gaze_time = 0
    gaze_duration_threshold = 2.0  # seconds
    
    while eye_tracking_active:
        try:
            _, frame = cam.read()
            if frame is None:
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = face_mesh.process(rgb_frame)
            landmark_points = output.multi_face_landmarks
            
            frame_h, frame_w, _ = frame.shape
            
            if landmark_points:
                landmarks = landmark_points[0].landmark
                
                # Eye landmarks for cursor control (right eye)
                for id, landmark in enumerate(landmarks[474:478]):
                    x = int(landmark.x * frame_w)
                    y = int(landmark.y * frame_h)
                    if id == 1:
                        screen_x = screen_w * landmark.x
                        screen_y = screen_h * landmark.y
                        
                        # Add to gaze history for stabilization
                        gaze_history.append((screen_x, screen_y))
                        if len(gaze_history) > max_history:
                            gaze_history.pop(0)
                        
                        # Calculate average gaze position
                        avg_x = sum([pos[0] for pos in gaze_history]) / len(gaze_history)
                        avg_y = sum([pos[1] for pos in gaze_history]) / len(gaze_history)
                        
                        gaze_coordinates = (avg_x, avg_y)
                        
                        # Check for prolonged gaze
                        current_time = time.time()
                        if current_time - last_gaze_time > gaze_duration_threshold:
                            select_text_at_gaze(gaze_coordinates)
                            last_gaze_time = current_time
                
                # Blink detection for click
                left_eye = [landmarks[145], landmarks[159]]
                right_eye = [landmarks[374], landmarks[386]]
                
                # Check for blink (both eyes closed)
                left_eye_closed = (left_eye[0].y - left_eye[1].y) < 0.008
                right_eye_closed = (right_eye[0].y - right_eye[1].y) < 0.008
                
                current_time = time.time()
                if left_eye_closed and right_eye_closed and (current_time - click_cooldown) > 2:
                    # Double blink detected - select text at current gaze position
                    select_text_at_gaze(gaze_coordinates)
                    click_cooldown = current_time
                    
        except Exception as e:
            print(f"Eye tracking error: {e}")
            continue
    
    cam.release()
    print("Eye tracking stopped")

def select_text_at_gaze(gaze_coords):
    """Select text based on gaze coordinates using actual OCR"""
    global selected_text, current_document_data
    
    if not current_document_data:
        return
    
    x, y = gaze_coords
    
    try:
        # Extract text from the region around gaze coordinates
        if current_document_data['file_type'] == 'pdf':
            # For PDFs, use the extracted text with positions
            text_data = current_document_data['text_data']
            screen_w, screen_h = pyautogui.size()
            
            # Convert screen coordinates to PDF coordinates (simplified)
            # This would need calibration based on how the PDF is displayed
            pdf_x = (x / screen_w) * 1000  # Assuming PDF width ~1000
            pdf_y = (y / screen_h) * 1000  # Assuming PDF height ~1000
            
            detected_text = find_text_at_position(text_data, pdf_x, pdf_y)
            
        else:
            # For images, use region-based OCR
            file_path = current_document_data['file_path']
            detected_text = extract_text_from_region(file_path, (x, y))
        
        if detected_text and len(detected_text.strip()) > 10:  # Minimum text length
            selected_text = detected_text
            print(f"Detected text at ({x}, {y}): {detected_text[:100]}...")
        else:
            selected_text = "No text detected in this area. Try looking at text more clearly."
            
    except Exception as e:
        print(f"Text selection error: {e}")
        selected_text = "Error detecting text in this region."

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        try:
            conn.execute(
                'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                (username, email, generate_password_hash(password))
            )
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'error')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/reader')
def reader():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('reader.html')

@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    global eye_tracking_active, tracking_thread
    
    if eye_tracking_active:
        return jsonify({'error': 'Eye tracking already active'}), 400
    
    eye_tracking_active = True
    tracking_thread = threading.Thread(target=start_eye_tracking)
    tracking_thread.daemon = True
    tracking_thread.start()
    
    return jsonify({'success': True, 'message': 'Eye tracking started'})

@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    global eye_tracking_active
    
    eye_tracking_active = False
    return jsonify({'success': True, 'message': 'Eye tracking stopped'})

@app.route('/get_gaze_data')
def get_gaze_data():
    global gaze_coordinates, selected_text
    
    data = {
        'gaze_x': gaze_coordinates[0],
        'gaze_y': gaze_coordinates[1],
        'selected_text': selected_text,
        'tracking_active': eye_tracking_active
    }
    
    # Clear selected text after reading
    text_to_return = selected_text
    selected_text = ""
    
    return jsonify({
        'gaze_x': data['gaze_x'],
        'gaze_y': data['gaze_y'],
        'selected_text': text_to_return,
        'tracking_active': data['tracking_active']
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_document_data
    
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process file based on type
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        if file_ext == 'pdf':
            # Convert PDF to image for display
            img_data = pdf_to_image(filepath)
            if img_data:
                img_b64 = base64.b64encode(img_data).decode('utf-8')
                # Extract text from PDF with positions
                text_data = extract_text_from_pdf(filepath)
                full_text = "\n".join([item['text'] for item in text_data])
                
                # Store document data for text selection
                current_document_data = {
                    'file_type': 'pdf',
                    'file_path': filepath,
                    'text_data': text_data,
                    'full_text': full_text
                }
            else:
                return jsonify({'error': 'Failed to process PDF'}), 500
        else:
            # For images, use the file directly
            with open(filepath, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
            # Extract text using Gemini
            full_text = process_image_with_genai(filepath)
            
            # Store document data for text selection
            current_document_data = {
                'file_type': 'image',
                'file_path': filepath,
                'full_text': full_text
            }
        
        session['current_file'] = filename
        session['extracted_text'] = full_text
        
        return jsonify({
            'success': True,
            'image_data': f"data:image/png;base64,{img_b64}",
            'filename': filename,
            'extracted_text': full_text[:500] + "..." if len(full_text) > 500 else full_text
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/process_text', methods=['POST'])
def process_text():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    text = data.get('text', '')
    target_lang = data.get('language', 'en')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Translate text
        translated = translator.translate(text, dest=target_lang)
        translated_text = translated.text
        
        # Convert to speech
        audio_filename = f"output_{session['user_id']}_{int(time.time())}.mp3"
        audio_path = text_to_speech(translated_text, target_lang, audio_filename)
        
        if audio_path:
            return jsonify({
                'success': True,
                'translated_text': translated_text,
                'audio_url': f'/audio/{audio_filename}'
            })
        else:
            return jsonify({'error': 'Failed to generate audio'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/audio/<filename>')
def get_audio(filename):
    return send_file(os.path.join(app.config['AUDIO_FOLDER'], filename))

@app.route('/logout')
def logout():
    global eye_tracking_active, current_document_data
    eye_tracking_active = False
    current_document_data = None
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    # Initialize database before running the app
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)