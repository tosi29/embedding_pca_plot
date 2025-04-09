#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from google import genai
from google.genai import types

# --- Configuration ---
API_RETRY_DELAY = 10  # Seconds to wait between API calls
DEFAULT_LABEL = "unknown"
TEXT_FILE_LABEL = "text_file_data"

# --- Helper Functions ---
def get_embedding(client, text):
    """Generates embedding for a given text using the Gemini API."""
    try:
        result = client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=text,
            config=types.EmbedContentConfig(task_type="CLUSTERING"),
        )
        time.sleep(API_RETRY_DELAY) # Respect API rate limits
        return result.embeddings[0].values
    except Exception as e:
        print(f"Error generating embedding for '{text[:50]}...': {e}")
        return None

def is_valid_embedding(embedding):
    """Checks if the provided embedding is a list of numbers."""
    return isinstance(embedding, list) and all(isinstance(n, (int, float)) for n in embedding)

# --- Main Logic ---
def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Generate PCA plot from text embeddings.")
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="Path to the input file (.txt or .json)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Path to the output HTML file (default: derived from input filename)"
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    # 2. Determine Output Path if not provided
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_pca.html")

    # 3. Initialize Gemini Client
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        exit(1)
    client = genai.Client(api_key=api_key)

    # 4. Initialize Data Lists
    texts = []
    embeddings = []
    labels = []

    # 5. Read and Process Input File
    print(f"Processing input file: {input_path}")
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        exit(1)

    file_extension = input_path.suffix.lower()

    try:
        if file_extension == ".txt":
            with open(input_path, "r", encoding="utf-8") as f:
                raw_texts = [line.strip() for line in f if line.strip()]
            print(f"Found {len(raw_texts)} non-empty lines in TXT file.")
            for text in raw_texts:
                embedding = get_embedding(client, text)
                if embedding:
                    texts.append(text)
                    embeddings.append(embedding)
                    labels.append(TEXT_FILE_LABEL) # Use a default label for txt files

        elif file_extension == ".json":
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON data must be a list of objects.")

            print(f"Found {len(data)} items in JSON file.")
            for item in data:
                if not isinstance(item, dict):
                    print("Warning: Skipping non-dictionary item in JSON list.")
                    continue

                text = item.get("text")
                if not text or not isinstance(text, str):
                    print("Warning: Skipping item with missing or invalid 'text' field.")
                    continue

                embedding = item.get("embedding")
                label = item.get("label", DEFAULT_LABEL) # Use default if label is missing

                if embedding and is_valid_embedding(embedding):
                    print(f"Using provided embedding for '{text[:30]}...'")
                    embeddings.append(embedding)
                else:
                    print(f"Generating embedding for '{text[:30]}...'")
                    embedding = get_embedding(client, text)
                    if not embedding:
                        print(f"Warning: Failed to get embedding for '{text[:30]}...'. Skipping.")
                        continue
                    embeddings.append(embedding)

                texts.append(text)
                labels.append(label)

        else:
            print(f"Error: Unsupported file extension '{file_extension}'. Please use .txt or .json.")
            exit(1)

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file: {input_path}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during file processing: {e}")
        exit(1)

    # 6. Data Validation
    if not texts or not embeddings or not labels:
        print("Error: No valid data found to process.")
        exit(1)
    if not (len(texts) == len(embeddings) == len(labels)):
        print("Error: Data inconsistency - lengths of texts, embeddings, and labels do not match.")
        # This should ideally not happen with the current logic, but good to check.
        exit(1)

    print(f"Successfully processed {len(texts)} items.")

    # 7. Perform PCA
    print("Performing PCA...")
    try:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(np.array(embeddings))
    except Exception as e:
        print(f"Error during PCA: {e}")
        exit(1)

    # 8. Generate Plot
    print("Generating plot...")
    try:
        fig = px.scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            color=labels,          # Use labels for color coding
            hover_name=texts,      # Show text on hover
            title=f'PCA of Embeddings from {input_path.name}',
            labels={'x': 'PCA Component 1', 'y': 'PCA Component 2', 'color': 'Label'}
        )
        fig.update_traces(textposition='top center') # Although hover_name is used, this doesn't hurt
    except Exception as e:
        print(f"Error during plot generation: {e}")
        exit(1)

    # 9. Save Plot
    print(f"Saving plot to: {output_path}")
    try:
        fig.write_html(output_path)
        print("Plot saved successfully.")
    except Exception as e:
        print(f"Error saving plot to HTML: {e}")
        exit(1)

if __name__ == "__main__":
    main()
