from flask import Flask, request, render_template, jsonify, redirect, url_for, session, send_from_directory
import os
import numpy as np
import pandas as pd
import librosa
import joblib
import tensorflow as tf
import parselmouth
from parselmouth.praat import call
from pydub import AudioSegment
import warnings
import base64
import json

# --- Basic Setup ---
app = Flask(__name__)
app.secret_key = 'a-very-secret-key-for-your-session'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration ---
class Config:
    SAMPLE_RATE = 44100
    N_MELS_CRNN = 128
    MAX_PAD_LEN_CRNN = 250
    N_MFCC_RF = 13
config = Config()

# --- Task Details Dictionary ---
TASK_DETAILS = {
    'task1_picture': {"title": "Picture Description"},
    'task2_memorize': {"title": "Word Memorization"},
    'task3_animals': {"title": "Category Fluency (Animals)"},
    'task4_tapping': {"title": "Finger Tapping"},
    'task4_memory_recall': {"title": "Memory Recall"},
    'task5_vowel': {"title": "Sustained Vowel"},
    'task6_articulation': {"title": "Pa-Ta-Ka Repetition"},
    'task6_rainbow': {"title": "Rainbow Passage Reading"},
    'task7_story': {"title": "Storytelling Recall"},
    'task8_emotion': {"title": "Emotion Recognition"},
}

# --- Load Models and Assets ---
# (Your model loading code is correct and does not need changes)
try:
    print("Loading models and assets...")
    crnn_model = tf.keras.models.load_model('crnn_model.h5')
    rf_model = joblib.load('random_forest_model.joblib')
    scaler = joblib.load('audio_scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    rf_feature_cols = joblib.load('rf_feature_columns.joblib')
    print("✅ All components loaded successfully!")
except Exception as e:
    print(f"❌ Error loading files: {e}")


# --- Feature Extraction & Prediction Logic (Unchanged) ---
# (Your run_prediction, extract_features_for_rf, etc. are correct)
def extract_features_for_rf(wav_path):
    features = {}
    try:
        y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE)
        sound = parselmouth.Sound(wav_path)
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        features['jitter_local'] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features['shimmer_local'] = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        features['hnr'] = call(harmonicity, "Get mean", 0, 0)
        pitch = sound.to_pitch(None, 75, 600)
        intensity = sound.to_intensity()
        features['mean_f0'] = call(pitch, "Get mean", 0, 0, "Hertz")
        features['std_dev_f0'] = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        features['mean_intensity'] = call(intensity, "Get mean", 0, 0, "energy")
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC_RF)
        for i in range(config.N_MFCC_RF):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                features[key] = 0
        return features
    except Exception:
        return None

def extract_spectrogram_for_crnn(wav_path):
    try:
        y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=config.N_MELS_CRNN)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        if mel_spec_db.shape[1] > config.MAX_PAD_LEN_CRNN:
            mel_spec_db = mel_spec_db[:, :config.MAX_PAD_LEN_CRNN]
        else:
            pad_width = config.MAX_PAD_LEN_CRNN - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return mel_spec_db
    except Exception:
        return None

def run_prediction(wav_path):
    rf_features = extract_features_for_rf(wav_path)
    crnn_spec = extract_spectrogram_for_crnn(wav_path)
    if rf_features is None or crnn_spec is None: return None
    rf_df = pd.DataFrame([rf_features], columns=rf_feature_cols)
    rf_scaled = scaler.transform(rf_df)
    rf_probs = rf_model.predict_proba(rf_scaled)[0]
    crnn_reshaped = np.expand_dims(crnn_spec, axis=0)
    crnn_probs = crnn_model.predict(crnn_reshaped, verbose=0)[0]
    ensemble_probs = (rf_probs + crnn_probs) / 2.0
    pred_idx = int(np.argmax(ensemble_probs))
    pred_label = label_encoder.classes_[pred_idx]
    confidence = float(ensemble_probs[pred_idx] * 100)
    return {
        'label': pred_label.capitalize(),
        'confidence': confidence,
        'probs': {label_encoder.classes_[i]: float(prob) for i, prob in enumerate(ensemble_probs)}
    }

# --- Main Routes ---
@app.route('/')
def index():
    session.clear()
    return render_template('index.html', TASK_DETAILS=TASK_DETAILS)

@app.route('/report')
def report_page():
    result = session.get('analysis_result')
    if not result: return redirect(url_for('index'))
    return render_template('report.html', result=result)

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files: return "No file part", 400
    file = request.files['audio']
    if file.filename == '': return "No selected file", 400
    in_path = os.path.join(UPLOAD_FOLDER, "uploaded_audio")
    file.save(in_path)
    wav_path = os.path.splitext(in_path)[0] + '.wav'
    try:
        sound = AudioSegment.from_file(in_path).set_frame_rate(config.SAMPLE_RATE)
        sound.export(wav_path, format='wav')
        prediction_result = run_prediction(wav_path)
        if prediction_result:
            session['analysis_result'] = prediction_result
            return redirect(url_for('report_page'))
        else:
            return "Error during feature extraction.", 500
    except Exception as e:
        return f"Processing failed: {str(e)}", 500
    finally:
        if os.path.exists(in_path): os.remove(in_path)
        if os.path.exists(wav_path): os.remove(wav_path)

# --- Routes for Guided Task System ---
@app.route('/start_session')
def start_session():
    session['task_results'] = {}
    return redirect(url_for('serve_task_statically', filename='task1.html'))

@app.route('/tasks/<path:filename>')
def serve_task_statically(filename):
    return send_from_directory('static/tasks', filename)

@app.route('/api/process_task_audio', methods=['POST'])
def process_task_audio():
    # (This function is correct from the last fix)
    task_id_str = request.form.get('task_id_str')
    audio_file = request.files.get('audio_blob')
    if not task_id_str or not audio_file:
        return jsonify({"status": "error", "message": "Missing data"}), 400
    
    temp_path = os.path.join(UPLOAD_FOLDER, f"{task_id_str}.webm")
    wav_path = os.path.join(UPLOAD_FOLDER, f"{task_id_str}.wav")
    try:
        audio_file.save(temp_path)
        sound = AudioSegment.from_file(temp_path).set_frame_rate(config.SAMPLE_RATE)
        sound.export(wav_path, format="wav")
        result = run_prediction(wav_path)
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)
        if os.path.exists(wav_path): os.remove(wav_path)
    
    if result:
        task_results = session.get('task_results', {})
        task_results[task_id_str] = result
        session['task_results'] = task_results
        return jsonify({"status": "success", "data": result})
    else:
        return jsonify({"status": "error", "message": "Prediction failed"}), 500

@app.route('/api/save_task_result', methods=['POST'])
def save_task_result():
    # (This function is correct)
    task_id_str = request.form.get('task_id_str')
    payload = request.form.get('payload')
    if not task_id_str or not payload:
        return jsonify({"status": "error", "message": "Missing data"}), 400
    
    task_results = session.get('task_results', {})
    task_results[task_id_str] = json.loads(payload)
    session['task_results'] = task_results
    return jsonify({"status": "success"})
    
@app.route('/generate_report')
def generate_report():
    # (This function is correct)
    task_results = session.get('task_results', {})
    if not task_results:
        return redirect(url_for('index'))

    labels = [res['label'] for res in task_results.values() if 'label' in res]
    if not labels:
        final_label = "Inconclusive"
        final_confidence = 0
    else:
        final_label = max(set(labels), key=labels.count)
        confidences = [res['confidence'] for res in task_results.values() if res.get('label') == final_label]
        final_confidence = sum(confidences) / len(confidences) if confidences else 0

    report_tasks = []
    for task_id, result in task_results.items():
        task_info = {
            "title": TASK_DETAILS.get(task_id, {}).get("title", task_id),
            "label": result.get("label", "N/A"),
            "confidence": result.get("confidence", 0)
        }
        report_tasks.append(task_info)

    final_report = {
        "label": final_label, "confidence": final_confidence, "tasks": report_tasks
    }
    last_audio_result = next((res for res in reversed(task_results.values()) if 'probs' in res), None)
    if last_audio_result:
        final_report['probs'] = last_audio_result['probs']

    session['analysis_result'] = final_report
    return redirect(url_for('report_page'))

# --- FIX: Add helper routes for static assets from unchanged task files ---
@app.route('/styles.css')
def serve_styles():
    return send_from_directory('static', 'styles.css')

@app.route('/app.js')
def serve_app_js():
    return send_from_directory('static', 'app.js')

@app.route('/assets/<path:subpath>')
def serve_assets(subpath):
    return send_from_directory('static/assets', subpath)
# --- End of Fix ---

if __name__ == '__main__':
    app.run(debug=True)