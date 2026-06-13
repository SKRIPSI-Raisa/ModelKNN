import os
import cv2
import time
import json
import pickle
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Define paths
MODEL_PATH = "knn_model.pkl"
DATA_PATH = "training_data.pkl"
METRICS_PATH = "metrics.json"

print("="*60)
print("     WasteSort - REAL-TIME CAMERA TESTING ENGINE (KNN)      ")
print("="*60)

# Check model files
missing_files = []
for p in [MODEL_PATH, DATA_PATH, METRICS_PATH]:
    if not os.path.exists(p):
        missing_files.append(p)

if missing_files:
    print(f"\n[ERROR] File model berikut tidak ditemukan: {', '.join(missing_files)}")
    print("Harap jalankan 'train.py' terlebih dahulu untuk melatih model dan menghasilkan file pickle.")
    print("="*60)
    input("\nTekan Enter untuk keluar...")
    exit(1)

# Try importing skimage features for LBP and HOG
try:
    from skimage.feature import local_binary_pattern, hog
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("\n[WARNING] 'scikit-image' tidak terinstall. Fitur LBP dan HOG mungkin tidak dapat diekstraksi secara lokal.")
    print("Silakan install dengan perintah: pip install scikit-image\n")

# Load model and metadata
print("Memuat file model...")
try:
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    print(f" -> {MODEL_PATH} berhasil dimuat.")
    CLF_LOADED = True
except Exception as e:
    print(f" -> Gagal memuat {MODEL_PATH} via Pickle (kemungkinan perbedaan versi scikit-learn): {e}")
    print(" -> Memasuki mode Fallback (Menggunakan perhitungan jarak manual NumPy).")
    CLF_LOADED = False

try:
    with open(DATA_PATH, "rb") as f:
        training_data = pickle.load(f)
    references = training_data["references"]
    train_features = np.array([ref["feature"] for ref in references])
    train_labels = np.array([ref["label"] for ref in references])
    feature_type = training_data["feature_type"]
    print(f" -> {DATA_PATH} berhasil dimuat (Fitur aktif: {feature_type.upper()}).")
except Exception as e:
    print(f"[ERROR] Gagal memuat {DATA_PATH}: {e}")
    input("\nTekan Enter untuk keluar...")
    exit(1)

try:
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    k_value = metrics.get("k_optimal", 19)
    model_accuracy = metrics.get("accuracy", 0.0) * 100
    print(f" -> {METRICS_PATH} berhasil dimuat (K-Optimal: {k_value}, Akurasi Tes: {model_accuracy:.2f}%).")
except Exception as e:
    k_value = 19
    model_accuracy = 0.0
    print(f" -> Gagal memuat/membaca {METRICS_PATH}, menggunakan default K={k_value}.")

# Helper feature extraction functions matching train.py
def extract_features(img_bgr, feat_type):
    # Resize to standard size (128x128)
    img_resized = cv2.resize(img_bgr, (128, 128))
    
    if feat_type == "color":
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
        hist_color = np.concatenate([hist_h, hist_s, hist_v]).flatten()
        sum_val = hist_color.sum()
        if sum_val > 0:
            hist_color /= sum_val
        return hist_color
        
    elif feat_type == "lbp":
        if not SKIMAGE_AVAILABLE:
            raise RuntimeError("scikit-image dibutuhkan untuk ekstraksi LBP.")
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        n_bins = int(lbp.max() + 1)
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist_lbp = hist_lbp.astype("float32")
        sum_lbp = hist_lbp.sum()
        if sum_lbp > 0:
            hist_lbp /= sum_lbp
        return hist_lbp
        
    elif feat_type == "hog":
        if not SKIMAGE_AVAILABLE:
            raise RuntimeError("scikit-image dibutuhkan untuk ekstraksi HOG.")
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        hog_feat = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
        return hog_feat
        
    elif feat_type == "combined":
        # Color
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
        hist_color = np.concatenate([hist_h, hist_s, hist_v]).flatten()
        sum_val = hist_color.sum()
        if sum_val > 0:
            hist_color /= sum_val
            
        # LBP
        if not SKIMAGE_AVAILABLE:
            raise RuntimeError("scikit-image dibutuhkan untuk ekstraksi LBP dalam mode Combined.")
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        n_bins = int(lbp.max() + 1)
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist_lbp = hist_lbp.astype("float32")
        sum_lbp = hist_lbp.sum()
        if sum_lbp > 0:
            hist_lbp /= sum_lbp
            
        combined = np.concatenate([hist_color, hist_lbp])
        return combined
    else:
        raise ValueError(f"Feature type '{feat_type}' tidak dikenali.")

# Initialize Camera
camera_idx = 0
cap = cv2.VideoCapture(camera_idx)

if not cap.isOpened():
    print(f"\n[WARNING] Tidak dapat membuka kamera pada index {camera_idx}.")
    print("Mencoba mencari index kamera lain...")
    for idx in range(1, 5):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            camera_idx = idx
            print(f" -> Berhasil terhubung ke kamera index: {camera_idx}")
            break
            
if not cap.isOpened():
    print("\n[ERROR] Kamera tidak terdeteksi pada sistem Anda.")
    print("Pastikan webcam sudah dicolokkan dan tidak sedang digunakan oleh aplikasi lain.")
    print("="*60)
    input("\nTekan Enter untuk keluar...")
    exit(1)

# Set resolution to HD if supported
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Performance & State Variables
fps = 0.0
fps_counter = 0
fps_start_time = time.time()

use_roi = True          # True: klasifikasi hanya bagian dalam kotak, False: full screen
show_hud = True         # Toggle visual HUD overlays
is_paused = False       # Pause video feed

# Color Palette (BGR)
COLOR_GREEN = (46, 204, 113)    # Organik
COLOR_RED = (231, 76, 60)       # Non-Organik
COLOR_WHITE = (236, 240, 241)
COLOR_GRAY = (149, 165, 166)
COLOR_CYAN = (241, 196, 15)      # Yellow/Cyan for UI highlights
COLOR_BG_BOX = (44, 62, 80)

print("\n[INFO] Kontrol Keyboard:")
print("  - [Q] : Keluar dari program")
print("  - [F] : Toggle Mode Deteksi (Kotak ROI / Layar Penuh)")
print("  - [P] : Pause / Resume deteksi")
print("  - [H] : Sembunyikan / Tampilkan HUD Info")
print("  - [S] : Simpan gambar klasifikasi saat ini")
print("="*60)
print("Memulai kamera feed. Menampilkan window...")

while True:
    if not is_paused:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame dari kamera.")
            break
            
        # Flip frame horizontally for natural mirror effect
        frame = cv2.flip(frame, 1)
        
    h, w, _ = frame.shape
    
    # Calculate crop/ROI dimensions (Center Box)
    box_size = min(h, w) // 2
    if box_size < 128:
        box_size = 128
        
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size
    
    # Extract Region for Classification
    if use_roi:
        roi = frame[y1:y2, x1:x2]
        classify_img = roi.copy()
    else:
        classify_img = frame.copy()
        
    prediction_label = "Menunggu..."
    confidence = 0.0
    organik_votes = 0
    non_organik_votes = 0
    neighbors_list = []
    
    # Run classification
    try:
        # Extract features from input
        feat = extract_features(classify_img, feature_type)
        
        if CLF_LOADED:
            # 1. Prediction using Scikit-Learn KNeighborsClassifier
            pred_class = clf.predict(feat.reshape(1, -1))[0]
            probs = clf.predict_proba(feat.reshape(1, -1))[0]
            
            # Map classes to probability index
            class_indices = list(clf.classes_)
            pred_idx = class_indices.index(pred_class)
            confidence = probs[pred_idx]
            prediction_label = str(pred_class)
            
            # Get neighbors info
            dists, idxs = clf.kneighbors(feat.reshape(1, -1), n_neighbors=k_value)
            dists = dists[0]
            idxs = idxs[0]
            
            # Count votes
            for rank_idx, idx in enumerate(idxs):
                ref = references[idx]
                dist = dists[rank_idx]
                neighbors_list.append({
                    "rank": rank_idx + 1,
                    "label": ref["label"],
                    "category": ref["category"],
                    "distance": dist
                })
                if ref["label"] == "Organik":
                    organik_votes += 1
                else:
                    non_organik_votes += 1
        else:
            # 2. Fallback: Pure NumPy Euclidean distance calculation
            diffs = train_features - feat
            distances = np.linalg.norm(diffs, axis=1)
            nearest_indices = np.argsort(distances)[:k_value]
            
            # Vote counting
            for rank_idx, idx in enumerate(nearest_indices):
                ref = references[idx]
                dist = distances[idx]
                neighbors_list.append({
                    "rank": rank_idx + 1,
                    "label": ref["label"],
                    "category": ref["category"],
                    "distance": dist
                })
                if ref["label"] == "Organik":
                    organik_votes += 1
                else:
                    non_organik_votes += 1
                    
            if organik_votes >= non_organik_votes:
                prediction_label = "Organik"
                confidence = organik_votes / k_value
            else:
                prediction_label = "Non-Organik"
                confidence = non_organik_votes / k_value
                
    except Exception as e:
        prediction_label = "Error Ekstraksi"
        confidence = 0.0
        
    # Calculate FPS
    fps_counter += 1
    if (time.time() - fps_start_time) > 1.0:
        fps = fps_counter / (time.time() - fps_start_time)
        fps_counter = 0
        fps_start_time = time.time()
        
    # Create Display Image
    display_frame = frame.copy()
    
    # 1. Draw target box / ROI if enabled
    box_color = COLOR_GREEN if prediction_label == "Organik" else (COLOR_RED if prediction_label == "Non-Organik" else COLOR_WHITE)
    
    if use_roi:
        # Draw corners for a premium futuristic scanner look
        len_corner = box_size // 5
        thickness = 4
        # Top-Left
        cv2.line(display_frame, (x1, y1), (x1 + len_corner, y1), box_color, thickness)
        cv2.line(display_frame, (x1, y1), (x1, y1 + len_corner), box_color, thickness)
        # Top-Right
        cv2.line(display_frame, (x2, y1), (x2 - len_corner, y1), box_color, thickness)
        cv2.line(display_frame, (x2, y1), (x2, y1 + len_corner), box_color, thickness)
        # Bottom-Left
        cv2.line(display_frame, (x1, y2), (x1 + len_corner, y2), box_color, thickness)
        cv2.line(display_frame, (x1, y2), (x1, y2 - len_corner), box_color, thickness)
        # Bottom-Right
        cv2.line(display_frame, (x2, y2), (x2 - len_corner, y2), box_color, thickness)
        cv2.line(display_frame, (x2, y2), (x2, y2 - len_corner), box_color, thickness)
        
        # Subtle dotted outline box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 1)
        
        # Text instruction inside target area if waiting/neutral
        if prediction_label not in ["Organik", "Non-Organik"]:
            cv2.putText(display_frame, "TEMPATKAN SAMPAH DI SINI", (x1 + 10, y1 + box_size//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
    else:
        # Full frame mode visual indicator (thin border around screen)
        cv2.rectangle(display_frame, (5, 5), (w-5, h-5), box_color, 2)
        
    # 2. Draw HUD Overlay
    if show_hud:
        # Background transparency panel for top banner
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (20, 28, 38), -1) # Dark Banner
        
        # Details panel on the left
        cv2.rectangle(overlay, (15, 110), (320, 310), (20, 28, 38), -1)
        
        # Apply transparency
        alpha = 0.75
        cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
        
        # Draw classification results (Top Banner)
        if prediction_label in ["Organik", "Non-Organik"]:
            # Status Text
            status_text = f"{prediction_label.upper()} ({confidence * 100:.1f}%)"
            color_text = COLOR_GREEN if prediction_label == "Organik" else COLOR_RED
            
            cv2.putText(display_frame, status_text, (25, 55),
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, color_text, 2, cv2.LINE_AA)
            
            # Draw a visual confidence progress bar
            bar_w = 400
            bar_h = 10
            bar_x = 25
            bar_y = 68
            # Background bar
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
            # Filled bar
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + int(bar_w * confidence), bar_y + bar_h), color_text, -1)
        else:
            cv2.putText(display_frame, "SCANNING...", (25, 55),
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, COLOR_CYAN, 2, cv2.LINE_AA)
            
        # Draw metadata in left panel
        cv2.putText(display_frame, "METADATA MODEL KNN", (30, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 1, cv2.LINE_AA)
        
        # Line details
        y_offset = 165
        metadata_lines = [
            f"FPS: {fps:.1f}",
            f"Fitur: {feature_type.upper()}",
            f"K Optimal: {k_value}",
            f"Akurasi Model: {model_accuracy:.1f}%",
            f"Mode: {'Kotak ROI' if use_roi else 'Layar Penuh'}",
            f"Votes - Org: {organik_votes} | Non: {non_organik_votes}"
        ]
        
        for line in metadata_lines:
            cv2.putText(display_frame, line, (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)
            y_offset += 22
            
        # Highlight current status if paused
        if is_paused:
            cv2.putText(display_frame, "PAUSED", (w - 120, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_CYAN, 2, cv2.LINE_AA)
            
        # Draw hotkey help (Bottom area)
        help_y = h - 25
        cv2.putText(display_frame, "[Q] Keluar  [F] Mode Deteksi  [P] Pause  [S] Screenshot  [H] Toggle HUD",
                    (25, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)
                    
        # Visual notification when screenshot is saved
        # We can implement a quick text flash or print to terminal
        
    # Render Frame
    cv2.imshow("WasteSort KNN - Camera Testing", display_frame)
    
    # Capture keystrokes
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        print("\nKeluar dari program testing kamera...")
        break
        
    elif key == ord('f') or key == ord('F'):
        use_roi = not use_roi
        print(f"[INFO] Mode klasifikasi diubah ke: {'KOTAK ROI' if use_roi else 'LAYAR PENUH'}")
        
    elif key == ord('h') or key == ord('H'):
        show_hud = not show_hud
        
    elif key == ord('p') or key == ord('P'):
        is_paused = not is_paused
        print(f"[INFO] Status Kamera: {'DITANGGUHKAN (PAUSED)' if is_paused else 'AKTIF (RESUMED)'}")
        
    elif key == ord('s') or key == ord('S'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{prediction_label.lower()}_{timestamp}.png"
        cv2.imwrite(filename, display_frame)
        print(f"[SUCCESS] Screenshot berhasil disimpan: {filename}")
        
        # Visual feedback on screen (draw brief text indicator)
        flash_frame = display_frame.copy()
        cv2.putText(flash_frame, "SCREENSHOT DISIMPAN!", (w//2 - 150, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_CYAN, 2, cv2.LINE_AA)
        cv2.imshow("WasteSort KNN - Camera Testing", flash_frame)
        cv2.waitKey(500)  # Show flash for 500ms

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Kamera ditutup. Selesai.")
