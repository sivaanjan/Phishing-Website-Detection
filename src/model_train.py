import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from urllib.parse import urlparse
import re
import os
import math

# --- CONFIGURATION ---
DATA_FILE = 'C:\\Users\\Siva\\OneDrive\\Desktop\\phishing\\phishing\\dataset.csv' 
MODEL_FILENAME = 'C:\\Users\\Siva\\OneDrive\\Desktop\\phishing\\phishing\\phishing_model.pkl'

# --- CONSTANTS ---
SENSITIVE_KEYWORDS = ['login', 'verify', 'account', 'security', 'bank', 'paypal', 'creditcard', 'update', 'confirm']

# 30 original features + 4 simulated features = 34 total features
ALL_FEATURE_NAMES = [
    'having_IPhaving_IP_Address', 'URLURL_Length', 'Shortining_Service', 'having_At_Symbol', 
    'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 
    'SSLfinal_State', 'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 
    'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 
    'Abnormal_URL', 'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 
    'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 
    'Links_pointing_to_page', 'Statistical_report',
    # --- SIMULATED NEW FEATURES ---
    'Punycode_Encoding', 
    'External_Form_Action',
    'High_Entropy_URL',
    'Sensitive_Word_Count' # FIXED: Added back to list
]

# --- FEATURE EXTRACTOR CLASS ---

class URLFeatureExtractor:
    def _safe_urlparse(self, url):
        try:
            return urlparse(url)
        except ValueError:
            return type('ParseResult', (object,), {
                'scheme': '', 'netloc': '', 'path': '', 
                'hostname': None, 'port': None
            })

    def _shannon_entropy(self, data):
        if not data: return 0
        frequencies = {}
        for char in data:
            frequencies[char] = frequencies.get(char, 0) + 1
        entropy = 0
        total_length = len(data)
        for freq in frequencies.values():
            p_x = freq / total_length
            entropy += p_x * math.log2(p_x)
        return -entropy

    def get_features(self, url):
        if not isinstance(url, str): return [1] * len(ALL_FEATURE_NAMES)
        
        url_with_protocol = url if url.startswith(('http://', 'https://')) else 'http://' + url 
        parsed_url = self._safe_urlparse(url_with_protocol)
        domain = parsed_url.netloc
        path = parsed_url.path
        
        feature_dict = {name: 1 for name in ALL_FEATURE_NAMES}

        # --- Calculate Lexical/Structural Features ---
        
        # 1. New Simulated Features
        feature_dict['Punycode_Encoding'] = -1 if 'xn--' in url.lower() else 1
        feature_dict['External_Form_Action'] = -1 if ('login' in url.lower() and 'https' not in url.lower()) else 1
        feature_dict['Sensitive_Word_Count'] = -1 if any(k in url.lower() for k in SENSITIVE_KEYWORDS) else 1
        
        entropy_score = self._shannon_entropy(domain + path)
        feature_dict['High_Entropy_URL'] = -1 if entropy_score > 4.5 else 1

        # 2. Original Features
        feature_dict['having_IPhaving_IP_Address'] = -1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain) else 1
        feature_dict['URLURL_Length'] = -1 if len(url) > 75 else (0 if len(url) > 54 else 1)
        feature_dict['Shortining_Service'] = -1 if any(s in url.lower() for s in ['bit.ly', 'tinyurl', 'goo.gl']) else 1
        feature_dict['having_At_Symbol'] = -1 if '@' in url else 1
        feature_dict['Prefix_Suffix'] = -1 if '-' in domain else 1
        feature_dict['SSLfinal_State'] = -1 if not url.startswith('https') else 1
        
        # 3. Features assumed Neutral (0)
        feature_dict['Redirect'] = 0 
        feature_dict['Abnormal_URL'] = 0 
        feature_dict['Domain_registeration_length'] = 0 
        feature_dict['age_of_domain'] = 0 
        feature_dict['web_traffic'] = 0 
        feature_dict['Page_Rank'] = 0 
        
        # --- Critical Heuristics ---
        is_highly_suspicious = (
            feature_dict['Sensitive_Word_Count'] == -1 or 
            feature_dict['Punycode_Encoding'] == -1 or
            feature_dict['High_Entropy_URL'] == -1
        )
        
        if is_highly_suspicious:
            feature_dict['DNSRecord'] = -1
            feature_dict['Links_in_tags'] = -1
        
        # Ensure correct ordering
        ordered_features = [feature_dict[name] for name in ALL_FEATURE_NAMES]
        return ordered_features

# --- TRAINING EXECUTION ---

def _standardize_column_names(df):
    df.columns = df.columns.str.strip()
    df_cols_lower = [c.lower() for c in df.columns]

    label_candidates = ['Result', 'result', 'Label', 'label', 'Target', 'target', 'Class', 'class']
    
    label_col_found = None
    for candidate in label_candidates:
        if candidate.lower() in df_cols_lower:
            original_col_name = df.columns[df_cols_lower.index(candidate.lower())]
            df = df.rename(columns={original_col_name: 'Label'})
            label_col_found = True
            break
            
    if not label_col_found:
        raise KeyError(f"Could not find a valid Label column. Actual headers: {list(df.columns)}")
    
    if 'index' in df.columns:
        df = df.drop('index', axis=1)
    
    return df

def generate_features_and_train():
    try:
        df_raw = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"File not found: {DATA_FILE}. Please check the path and filename.")
        return
    
    try:
        df = _standardize_column_names(df_raw)
    except KeyError as e:
        print(f"CRITICAL ERROR in data preparation: {e}")
        return
        
    # 2. Add Simulated Features for training data consistency
    # We map new features to existing related features to give the model signals during training
    df['Punycode_Encoding'] = df['having_At_Symbol'].apply(lambda x: -1 if x == -1 else 1)
    df['External_Form_Action'] = df['SFH'].apply(lambda x: -1 if x == -1 else 1)
    df['High_Entropy_URL'] = df['URLURL_Length'].apply(lambda x: -1 if x == -1 else 1)
    # Sensitive Word Count simulation: often correlated with URL length (longer URLs have more words)
    df['Sensitive_Word_Count'] = df['URLURL_Length'].apply(lambda x: -1 if x == -1 else 1)

    # 3. Select Features
    try:
        X = df[ALL_FEATURE_NAMES].values
    except KeyError as e:
        print(f"ERROR: Missing a required feature column in your dataset: {e}")
        return

    y = df['Label'].apply(lambda x: 1 if x == 1 else 0).values

    # 4. Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining on {len(X_train)} samples with {X.shape[1]} features...")
    
    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # 5. Evaluation & Save
    y_pred = model.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    metrics_to_save = {
        'phishing_recall': report_dict['0']['recall'],
        'phishing_precision': report_dict['0']['precision'],
        'overall_accuracy': report_dict['accuracy']
    }
    
    with open('C:\\Users\\Siva\\OneDrive\\Desktop\\phishing\\phishing\\project_metrics.pkl', 'wb') as f:
        pickle.dump(metrics_to_save, f)
        
    print(f"Model trained successfully. Test Accuracy: {report_dict['accuracy']:.4f}")
    
    with open(MODEL_FILENAME, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved successfully to: {MODEL_FILENAME}")

if __name__ == '__main__':
    generate_features_and_train()