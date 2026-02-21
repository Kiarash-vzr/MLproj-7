import pandas as pd
import re

import nltk

# Update these lines at the beginning of your code:
# English: Added 'punkt_tab' to avoid the LookupError in newer NLTK versions
nltk.download('punkt')
nltk.download('punkt_tab') # This is the missing piece
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Download required NLTK resources ---
# These datasets help NLTK recognize punctuation and common English stopwords
nltk.download('punkt')
nltk.download('stopwords')

def clean_text_for_absa(text):
    """
    This function cleans a single sentence by removing noise
    and keeping meaningful words only.
    """
    # 1. Convert text to lowercase
    text = str(text).lower()
    
    # 2. Remove punctuation and special characters using Regular Expressions
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Tokenize the text (split into words)
    tokens = word_tokenize(text)
    
    # 4. Filter out stopwords (words like 'a', 'the', 'is')
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Join words back into a clean string
    return " ".join(cleaned_tokens)

# --- Load the extracted data ---
# Replace 'restaurant_cleaned_data.csv' with your actual filename if it's different
df = pd.read_csv('restaurant_cleaned_data.csv')

# --- Apply the preprocessing function ---
# We apply the cleaning process to the 'text' column
df['processed_text'] = df['text'].apply(clean_text_for_absa)

# --- Save the final preprocessed file ---
# This file will be the direct input for our Deep Learning models (Phase 2)
df.to_csv('restaurant_preprocessed_final.csv', index=False)

# Display a quick comparison
print("Sample of Preprocessing results:")
print(df[['text', 'processed_text']].head())
print("\n✅ Preprocessing finished! File 'restaurant_preprocessed_final.csv' is ready.")