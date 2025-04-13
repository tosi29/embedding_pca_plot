from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN # Import HDBSCAN
# CountVectorizer is no longer explicitly needed here if relying on sentence-transformer's internal tokenizer
# from sklearn.feature_extraction.text import CountVectorizer
# MeCab and ipadic are likely handled internally by the Japanese sentence-transformer model

# --- Configuration ---
# Select a Sentence Transformer model suitable for Japanese/multilingual
# Using a Japanese-specific model as requested
EMBEDDING_MODEL_NAME = 'sonoisa/sentence-bert-base-ja-mean-tokens-v2'
INPUT_FILE = 'input.txt'
OUTPUT_HTML_FILE = 'bertopic_japanese_topics.html'

# --- MeCab Tokenizer Setup (Removed) ---
# The Japanese sentence-transformer model ('sonoisa/sentence-bert-base-ja-mean-tokens-v2')
# typically handles tokenization internally, often using MeCab via fugashi.
# Explicit MeCab setup and CountVectorizer tokenizer are usually not needed.

# --- Load Data ---
print(f"Loading documents from {INPUT_FILE}...")
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        documents = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(documents)} documents.")
    if not documents:
        print("Input file is empty or contains no valid lines.")
        exit(1)
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILE}' not found.")
    exit(1)
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

# --- BERTopic Setup ---
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
# The SentenceTransformer library will load the specified Japanese model.
# This model likely includes or requires a suitable tokenizer (like MeCab via fugashi).
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# print("Setting up CountVectorizer with Japanese tokenizer...") # Removed
# vectorizer = CountVectorizer(tokenizer=japanese_tokenizer) # Removed

print("Initializing BERTopic model...")
# When using a language-specific sentence transformer, BERTopic often doesn't
# require an explicit vectorizer_model, as tokenization is handled internally.
# Setting language='japanese' is appropriate.
# Create an HDBSCAN model instance with the desired min_cluster_size
hdbscan_model = HDBSCAN(min_cluster_size=3, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

topic_model = BERTopic(
    embedding_model=embedding_model,
    hdbscan_model=hdbscan_model, # Pass the HDBSCAN instance here
    language="japanese", # Set specifically to Japanese
    # min_cluster_size=3, # Removed incorrect parameter placement
    calculate_probabilities=True,
    verbose=True,
    nr_topics=None # Disable automatic topic reduction for small datasets
)

# --- Run BERTopic ---
print("Fitting BERTopic model and transforming documents...")
topics, probs = topic_model.fit_transform(documents)

# --- Display Results ---
print("\n--- BERTopic Results ---")
print(f"Found {len(topic_model.get_topic_info()) - 1} topics (excluding outliers).") # -1 for the outlier topic -1

# Print topic information (Top words, Count)
print("\nTopic Info:")
print(topic_model.get_topic_info())

# Print the top words for the first few topics
num_topics_to_show = min(5, len(topic_model.get_topic_info()) -1)
print(f"\nTop words for the first {num_topics_to_show} topics:")
for topic_id in range(num_topics_to_show):
    words = topic_model.get_topic(topic_id)
    print(f"Topic {topic_id}: {words}")

# --- Visualize Topics (Optional) ---
try:
    print(f"\nGenerating visualization HTML: {OUTPUT_HTML_FILE}...")
    fig = topic_model.visualize_topics()
    fig.write_html(OUTPUT_HTML_FILE)
    print(f"Visualization saved to {OUTPUT_HTML_FILE}")
except Exception as e:
    print(f"Could not generate visualization: {e}")

print("\nBERTopic analysis complete.")
