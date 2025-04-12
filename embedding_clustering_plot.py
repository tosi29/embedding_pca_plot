#!/usr/bin/env python3
import argparse
import json
import os
import time
import textwrap
from pathlib import Path
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA # Keep for now, maybe remove later if not needed elsewhere, but safer to keep
import umap
import hdbscan
from bertopic import BERTopic # Added for BERTopic
from google import genai
from google.genai import types

# --- Configuration ---
API_RETRY_DELAY = 10  # Seconds to wait between API calls
DEFAULT_LABEL = "unknown"
TEXT_FILE_LABEL = "text_file_data"
BERTOPIC_OUTLIER_LABEL = "Outliers -1" # Label for BERTopic outliers

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
    parser = argparse.ArgumentParser(description="Generate UMAP+HDBSCAN plot from text embeddings.") # Updated description
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
        # Updated default output filename
        output_path = input_path.with_name(f"{input_path.stem}_umap_hdbscan.html")

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
    details_texts = []
    original_texts = [] # Store original, unwrapped text for BERTopic

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
                    original_texts.append(text) # Store original text
                    wrapped_text = wrap_text(text) # Wrap for hover
                    texts.append(wrapped_text)
                    embeddings.append(embedding)
                    labels.append(TEXT_FILE_LABEL)
                    details_texts.append("")

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

                original_texts.append(text) # Store original text
                wrapped_text = wrap_text(text) # Wrap for hover
                wrapped_details = wrap_text(details) # Wrap for hover

                texts.append(wrapped_text)
                labels.append(label)
                details_texts.append(wrapped_details)

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
    embeddings_array = np.array(embeddings) # Convert embeddings list to numpy array

    # 7. Perform BERTopic Analysis
    print("Performing BERTopic analysis...")
    try:
        # Configure UMAP and HDBSCAN for BERTopic (can customize parameters)
        # Note: These models are used *internally* by BERTopic for topic generation.
        # We'll run a separate UMAP later just for 2D visualization.
        umap_model_bt = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        # Adjust min_cluster_size as needed for your data
        hdbscan_model_bt = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

        # Initialize BERTopic
        topic_model = BERTopic(
            umap_model=umap_model_bt,
            hdbscan_model=hdbscan_model_bt,
            embedding_model=None, # We provide pre-computed embeddings
            verbose=True
        )

        # Fit BERTopic model using original texts and pre-calculated embeddings
        topic_ids, _ = topic_model.fit_transform(original_texts, embeddings=embeddings_array)

        # Convert integer topic IDs to string labels for plotting
        bertopic_labels_str = [f"Topic {label}" if label != -1 else BERTOPIC_OUTLIER_LABEL for label in topic_ids]
        num_topics = len(set(topic_ids)) - (1 if -1 in topic_ids else 0)
        num_noise = np.sum(np.array(topic_ids) == -1)
        print(f"BERTopic found {num_topics} topics and {num_noise} outliers.")

        # Print Topic Info (Optional but helpful)
        print("\n--- BERTopic Info ---")
        print(topic_model.get_topic_info())
        print("\n--- Top Keywords per Topic ---")
        for topic_id in sorted(topic_model.get_topic_info()['Topic'].unique()):
             if topic_id == -1: continue # Skip outliers for keyword display
             keywords = topic_model.get_topic(topic_id)
             print(f"Topic {topic_id}: {keywords}")
        print("----------------------\n")


    except Exception as e:
        print(f"Error during BERTopic analysis: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # 8. Perform UMAP Dimensionality Reduction *for Plotting*
    print("Performing UMAP for 2D visualization...")
    try:
        # Use a separate UMAP instance just for 2D plotting
        reducer_plot = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, metric='cosine')
        umap_plot_result = reducer_plot.fit_transform(embeddings_array)
    except Exception as e:
        print(f"Error during UMAP for plotting: {e}")
        exit(1)


    # 9. Prepare Data for Plotting
    print("Preparing data for plotting...")
    # Data is aligned: umap_plot_result, bertopic_labels_str, labels (original), texts (wrapped), details_texts

    # 10. Generate Plot with UMAP results and BERTopic labels
    print("Generating plot...")
    try:
        fig = px.scatter(
            x=umap_plot_result[:, 0],
            y=umap_plot_result[:, 1],
            color=bertopic_labels_str, # Color by BERTopic label
            custom_data=[texts, details_texts, labels, bertopic_labels_str], # Add BERTopic label to custom data
            title=f'BERTopic Clustering (Visualized with UMAP) from {input_path.name}', # Updated title
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'color': 'BERTopic Topic'} # Updated labels
        )
        # Define the updated hover template
        hovertemplate = (
            f"<b>{args.json_text_field}:</b><br>" "%{customdata[0]}<br><br>"
            f"<b>{args.json_details_field}:</b><br>" "%{customdata[1]}<br><br>"
            f"<b>Original {args.json_label_field}:</b> " "%{customdata[2]}<br>" # Keep original label info
            f"<b>BERTopic Topic:</b> " "%{customdata[3]}<br>" # Show BERTopic Topic ID/Label
            "<extra></extra>" # Hide the trace info
        )
        fig.update_traces(hovertemplate=hovertemplate)
        # Optional: Explicitly color outliers gray
        # This requires mapping topic labels back to colors. Plotly might do this well enough.
        # Example:
        # color_map = {label: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        #              for i, label in enumerate(sorted(set(bertopic_labels_str) - {BERTOPIC_OUTLIER_LABEL}))}
        # color_map[BERTOPIC_OUTLIER_LABEL] = 'lightgrey'
        # fig.update_traces(marker=dict(color=[color_map[label] for label in bertopic_labels_str]))

    except Exception as e:
        print(f"Error during plot generation: {e}")
        exit(1)

    # 11. Save Plot (Index adjusted)
    print(f"Saving plot to: {output_path}")
    try:
        fig.write_html(output_path, include_plotlyjs='cdn' )
        print("Plot saved successfully.")
    except Exception as e:
        print(f"Error saving plot to HTML: {e}")
        exit(1)

if __name__ == "__main__":
    main()
