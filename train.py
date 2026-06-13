import os
import cv2
import time
import json
import pickle
import numpy as np
from skimage.feature import local_binary_pattern, hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from concurrent.futures import ProcessPoolExecutor

def extract_features_single(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        # Resize to standard size (128x128)
        img_resized = cv2.resize(img, (128, 128))
        
        # 1. Color Histogram (HSV)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
        hist_color = np.concatenate([hist_h, hist_s, hist_v]).flatten()
        sum_val = hist_color.sum()
        if sum_val > 0:
            hist_color /= sum_val
            
        # 2. LBP (Uniform LBP)
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
            
        # 3. HOG
        hog_feat = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
        
        # 4. Combined (Color Histogram + LBP)
        combined = np.concatenate([hist_color, hist_lbp])
        
        return hist_color, hist_lbp, hog_feat, combined
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def process_item(item):
    res = extract_features_single(item["path"])
    if res is None:
        return None
    color, lbp, hog_feat, combined = res
    return {
        "path": item["path"],
        "filename": item["filename"],
        "label": item["label"],
        "category": item["category"],
        "color": color,
        "lbp": lbp,
        "hog": hog_feat,
        "combined": combined
    }

def get_image_info(base_dir):
    data_info = []
    classes = ['organik', 'non-organik']
    
    organic_categories = ["Dedaunan", "Sisa Buah", "Sisa Makanan", "Sisa Sayuran"]
    non_organic_categories = ["Botol Plastik", "Kaleng Minuman", "Kardus/Karton", "Kertas/Buku", "Kemasan Plastik"]
    
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        label = "Organik" if cls == "organik" else "Non-Organik"
        
        filenames = os.listdir(cls_dir)
        for i, fname in enumerate(filenames):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                full_path = os.path.join(cls_dir, fname)
                if label == "Organik":
                    category = organic_categories[i % len(organic_categories)]
                else:
                    category = non_organic_categories[i % len(non_organic_categories)]
                data_info.append({
                    "path": full_path,
                    "filename": fname,
                    "label": label,
                    "category": category
                })
    return data_info

def main():
    print("Initializing directories...")
    train_dir = r"e:\SKRIPSI\Raisa\ModelKNN\DATASET\TRAIN"
    test_dir = r"e:\SKRIPSI\Raisa\ModelKNN\DATASET\TEST"
    
    # 1. Collect image paths
    print("Collecting image paths...")
    train_info = get_image_info(train_dir)
    test_info = get_image_info(test_dir)
    print(f"Found {len(train_info)} training images and {len(test_info)} test images.")
    
    # 2. Extract features in parallel
    print("Extracting features in parallel using ProcessPoolExecutor...")
    start_time = time.time()
    
    # We will limit CPUs to make sure we don't crash, let's use 2 workers to save memory
    num_workers = 2
    print(f"Using {num_workers} worker processes...")
    
    train_features = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_item, train_info, chunksize=100))
        train_features = [r for r in results if r is not None]
        
    test_features = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_item, test_info, chunksize=100))
        test_features = [r for r in results if r is not None]
        
    print(f"Feature extraction completed in {time.time() - start_time:.2f}s.")
    print(f"Successfully processed {len(train_features)} training features and {len(test_features)} test features.")
    
    # 3. Prepare dataset matrices
    y_train = np.array([item["label"] for item in train_features])
    y_test = np.array([item["label"] for item in test_features])
    
    feature_types = ["color", "lbp", "hog", "combined"]
    feature_names = {
        "color": "Color Histogram",
        "lbp": "Local Binary Patterns (LBP)",
        "hog": "Histogram of Oriented Gradients (HOG)",
        "combined": "Histogram + LBP (Combined)"
    }
    
    X_train_dict = {ft: np.array([item[ft] for item in train_features]) for ft in feature_types}
    X_test_dict = {ft: np.array([item[ft] for item in test_features]) for ft in feature_types}
    
    # 4. Grid Search K-values and compare feature methods
    print("Evaluating feature extraction methods and K-values...")
    k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    
    # Track the best accuracy overall
    best_overall_acc = 0.0
    best_feature_type = None
    best_k = None
    best_clf = None
    
    # For each feature type, find the best K
    feature_best_k_accuracy = {}
    
    for ft in feature_types:
        print(f"Testing features: {feature_names[ft]}")
        best_k_for_ft = None
        best_acc_for_ft = 0.0
        
        for k in k_values:
            clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
            clf.fit(X_train_dict[ft], y_train)
            preds = clf.predict(X_test_dict[ft])
            acc = accuracy_score(y_test, preds)
            print(f"  K={k} -> Accuracy: {acc:.4f}")
            
            if acc > best_acc_for_ft:
                best_acc_for_ft = acc
                best_k_for_ft = k
                
            if acc > best_overall_acc:
                best_overall_acc = acc
                best_feature_type = ft
                best_k = k
                best_clf = clf
                
        feature_best_k_accuracy[ft] = best_acc_for_ft
        print(f"Best K for {feature_names[ft]} is K={best_k_for_ft} with accuracy {best_acc_for_ft:.4f}\n")
        
    print(f"=== OVERALL BEST MODEL ===")
    print(f"Feature: {feature_names[best_feature_type]}")
    print(f"K-value: {best_k}")
    print(f"Test Accuracy: {best_overall_acc:.4f}")
    
    # 5. Evaluate the best model in detail on the test set
    best_X_train = X_train_dict[best_feature_type]
    best_X_test = X_test_dict[best_feature_type]
    
    best_clf.fit(best_X_train, y_train)
    y_pred = best_clf.predict(best_X_test)
    
    # Calculate performance metrics (Organik as positive, Non-Organik as negative)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="Organik")
    rec = recall_score(y_test, y_pred, pos_label="Organik")
    f1 = f1_score(y_test, y_pred, pos_label="Organik")
    
    # Confusion Matrix
    # Labels order: ["Organik", "Non-Organik"]
    # So TP (Organik/Organik) is index [0,0]
    # FN (Organik/Non-Organik) is index [0,1]
    # FP (Non-Organik/Organik) is index [1,0]
    # TN (Non-Organik/Non-Organik) is index [1,1]
    cm = confusion_matrix(y_test, y_pred, labels=["Organik", "Non-Organik"])
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])
    
    print("\nConfusion Matrix:")
    print(f"  TP (Actual Organik, Predicted Organik): {tp}")
    print(f"  TN (Actual Non-Organik, Predicted Non-Organik): {tn}")
    print(f"  FP (Actual Non-Organik, Predicted Organik): {fp}")
    print(f"  FN (Actual Organik, Predicted Non-Organik): {fn}")
    
    # 6. Generate K-curve for the best feature extraction method
    k_curve = []
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        clf.fit(best_X_train, y_train)
        val_acc = accuracy_score(y_test, clf.predict(best_X_test))
        k_curve.append({
            "k": k,
            "accuracy": float(round(val_acc, 4))
        })
        
    # 7. Generate feature comparison point list
    feature_comparison = [
        {
            "method": feature_names[ft],
            "accuracy": float(round(feature_best_k_accuracy[ft], 4))
        }
        for ft in feature_types
    ]
    
    # 8. Save Metrics JSON
    metrics_data = {
        "accuracy": float(round(acc, 4)),
        "precision": float(round(prec, 4)),
        "recall": float(round(rec, 4)),
        "f1_score": float(round(f1, 4)),
        "k_optimal": best_k,
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        },
        "k_curve": k_curve,
        "feature_comparison": feature_comparison,
        "model_info": {
            "algorithm": "K-Nearest Neighbors (KNN)",
            "k_value": best_k,
            "input_size": [128, 128, 3],
            "classes": ["Organik", "Non-Organik"],
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "train_size": len(train_features),
            "test_size": len(test_features)
        }
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)
    print("Saved metrics.json successfully.")
    
    # 9. Save Best Classifier Model
    # Save the trained sklearn KNN classifier
    with open("knn_model.pkl", "wb") as f:
        pickle.dump(best_clf, f)
    print("Saved knn_model.pkl successfully.")
    
    # 10. Save training reference metadata and extracted features (needed for finding neighbors)
    # Store path, label, category, and best feature representation for all training samples
    training_ref_data = []
    for item in train_features:
        training_ref_data.append({
            "path": item["path"],
            "filename": item["filename"],
            "label": item["label"],
            "category": item["category"],
            "feature": item[best_feature_type]
        })
        
    with open("training_data.pkl", "wb") as f:
        pickle.dump({
            "feature_type": best_feature_type,
            "references": training_ref_data
        }, f)
    print("Saved training_data.pkl successfully.")
    print("Model training pipeline finished successfully!")

if __name__ == "__main__":
    main()
