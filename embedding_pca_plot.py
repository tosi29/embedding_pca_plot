#!/usr/bin/env python3
import argparse
import json
import os
import time
import textwrap
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

def wrap_text(text, width=40):
    """Wraps text to a specified width, replacing newlines with <br>."""
    if not text: # Handle empty or None text
        return ""
    # Use textwrap.fill which handles existing newlines properly
    # Replace the generated newlines with <br> for HTML display
    return textwrap.fill(text, width=width).replace('\n', '<br>')

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
        "--json-text-field",
        type=str,
        default="text",
        help="Name of the field containing the text in the JSON input (default: text)"
    )
    parser.add_argument(
        "--json-embedding-field",
        type=str,
        default="embedding",
        help="Name of the field containing the embedding in the JSON input (default: embedding)"
    )
    parser.add_argument(
        "--json-label-field",
        type=str,
        default="label",
        help="Name of the field containing the label in the JSON input (default: label)"
    )
    parser.add_argument(
        "--json-details-field",
        type=str,
        default="details",
        help="Name of the field containing supplementary details in the JSON input (default: details)"
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
    details_texts = [] # Initialize list for details

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
                    # Wrap text before appending
                    wrapped_text = wrap_text(text)
                    texts.append(wrapped_text)
                    embeddings.append(embedding)
                    labels.append(TEXT_FILE_LABEL) # Use a default label for txt files
                    details_texts.append("") # Add empty string for details in txt files

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

                text = item.get(args.json_text_field)
                if not text or not isinstance(text, str):
                    print(f"Warning: Skipping item with missing or invalid '{args.json_text_field}' field.")
                    continue

                embedding = item.get(args.json_embedding_field)
                label = item.get(args.json_label_field, DEFAULT_LABEL) # Get label or default
                # Ensure label is a string, even if the field value was null/None
                if label is None:
                    label = DEFAULT_LABEL

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

                details = item.get(args.json_details_field, "")
                if not isinstance(details, str):
                    print(f"Warning: Details field '{args.json_details_field}' for '{text[:30]}...' is not a string. Using empty string.")
                    details = ""

                # Wrap text and details before appending
                wrapped_text = wrap_text(text)
                wrapped_details = wrap_text(details)

                texts.append(wrapped_text)
                labels.append(label)
                details_texts.append(wrapped_details) # Append wrapped details text

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

    # 8. Prepare Data for Plotting (Sort by Label for Legend Order)
    print("Preparing data for plotting (sorting by label)...")
    # Combine all data points into a list of tuples
    plot_data = list(zip(
        pca_result[:, 0],
        pca_result[:, 1],
        labels,
        texts,
        details_texts
    ))
    # Sort the data based on the label (3rd element, index 2)
    plot_data_sorted = sorted(plot_data, key=lambda item: item[2])

    # Unzip the sorted data back into separate lists/arrays
    sorted_x = [item[0] for item in plot_data_sorted]
    sorted_y = [item[1] for item in plot_data_sorted]
    sorted_labels = [item[2] for item in plot_data_sorted]
    sorted_texts = [item[3] for item in plot_data_sorted]
    sorted_details = [item[4] for item in plot_data_sorted]

    # 9. Generate Plot with Sorted Data
    print("Generating plot...")
    try:
        fig = px.scatter(
            x=sorted_x,
            y=sorted_y,
            color=sorted_labels,   # Use sorted labels for color coding and legend order
            custom_data=[sorted_texts, sorted_details, sorted_labels], # Pass sorted custom data
            title=f'PCA of Embeddings from {input_path.name}',
            labels={'x': 'PCA Component 1', 'y': 'PCA Component 2', 'color': 'Label'}
        )
        # Define the hover template (no max-width needed as text is pre-wrapped)
        hovertemplate = (
            f"<b>{args.json_text_field}:</b><br>" "%{customdata[0]}<br><br>"
            f"<b>{args.json_details_field}:</b><br>" "%{customdata[1]}<br><br>"
            f"<b>{args.json_label_field}:</b> " "%{customdata[2]}<br>"
            "<extra></extra>" # Hide the trace info
        )
        fig.update_traces(hovertemplate=hovertemplate)
    except Exception as e:
        print(f"Error during plot generation: {e}")
        exit(1)

    # 10. Save Plot
    print(f"Saving plot to: {output_path}")
    try:
        fig.write_html(output_path, include_plotlyjs='cdn' )
        print("Plot saved successfully.")
    except Exception as e:
        print(f"Error saving plot to HTML: {e}")
        exit(1)

if __name__ == "__main__":
    main()
