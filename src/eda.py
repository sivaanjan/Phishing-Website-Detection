import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# UPDATED: Using the absolute path specified by the user
DATA_FILE = 'C:\Users\Siva\OneDrive\Desktop\phishing\phishing\dataset.csv'

def load_data(file_path):
    """Loads the dataset."""
    try:
        # Note: Added 'low_memory=False' as large CSV files can cause DtypeWarning
        df = pd.read_csv(file_path, low_memory=False)
        return df.drop('index', axis=1, errors='ignore') # Drop the 'index' column if present
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the data: {e}")
        return None

def run_eda(df):
    """Performs key EDA visualizations."""
    print("--- Dataset Head ---")
    print(df.head())
    print("\n--- Dataset Info ---")
    print(df.info())
    print("\n--- Value Counts for Target (Result) ---")
    # Assuming Result: 1 (Legitimate), -1 (Phishing)
    print(df['Result'].value_counts()) 

    # 1. Target Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Result', data=df)
    plt.title('Distribution of Phishing vs. Legitimate Websites')
    plt.xlabel('Result (-1: Phishing, 1: Legitimate)')
    plt.show()

    # 2. Correlation Heatmap
    # Check correlation between features and target
    correlation_matrix = df.corr()
    
    plt.figure(figsize=(18, 16))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Feature Correlation Matrix')
    plt.show()
    

    # 3. Individual Feature Analysis (Example: SSLfinal_State)
    plt.figure(figsize=(10, 5))
    sns.countplot(x='SSLfinal_State', hue='Result', data=df)
    plt.title('SSL Final State vs. Result')
    plt.xlabel('SSLfinal_State (-1: Bad, 0: Neutral, 1: Good)')
    plt.legend(title='Result', labels=['Phishing (-1)', 'Legitimate (1)'])
    plt.show()

if __name__ == '__main__':
    data = load_data(DATA_FILE)
    if data is not None:
        run_eda(data)