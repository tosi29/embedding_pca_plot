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

# --- Topic Name Mapping ---
# トピック名をカスタムしたい場合は手動で定義する
TOPIC_NAME_MAPPING = {
    0: "危機下のリーダーシップ",
    1: "ジェンダー観と社会的構築",
    2: "歴史解釈と価値観の形成",
    3: "競争における戦略と戦術",
    4: "技術革新と社会・権力構造の変化",
    5: "教育・組織による知識と文化の継承",
    6: "失敗の構造と過信のリスク",
    7: "目標達成のための長期的努力と精神力",
    8: "組織・国家の変革と環境適応",
    9: "国際関係と安全保障のジレンマ",
    10: "民主主義の課題と権威主義",
    11: "統治システム：中央集権と地方分権",
    12: "国民・民族アイデンティティの形成と排他性",
    13: "社会問題と福祉国家の役割",
    14: "権力闘争と維持のための非情な手段",
    15: "人間の認識と信念の枠組み",
    16: "極限状況における非人間性と悲劇",
    17: "外部の脅威と組織の結束・崩壊",
    18: "思想・理想と政治的現実",
    19: "権威の源泉（伝統・武力・経済力など）",
    20: "社会改革の困難性と「Jカーブ効果」",
    21: "内部対立による組織の脆弱化",
    22: "時代を先取りする革新と再評価",
    23: "実力・経済力による影響力の獲得",
    24: "文脈依存的な評価と意味の相対性",
    25: "情報伝達技術と社会・権威の変革",
    26: "大規模変動（環境・社会）の複雑性と予期せぬ影響",
    27: "物語（ナラティブ）と象徴の政治的利用",
    28: "統治の理念（権力分立・一般意思）",
    29: "無限成長パラダイムへの批判と持続可能性",
    30: "抽象概念とシンボルの文化的発展",
    31: "承認の欠如と人格形成への影響",
    32: "文化の融合（シンクレティズム）と変容",
    33: "社会的危機と権威主義への傾倒",
    34: "外部リソースの活用と組織の発展",
    35: "効果的な学習と複雑な文脈の重要性",
    36: "リーダーへの依存と後継者問題のリスク",
    37: "使命感に基づく改革運動と対抗勢力",
    38: "科学の権威と差別の正当化",
    39: "可能性の認識と挑戦を促す心理",
    40: "常識と価値観の歴史的・社会的相対性",
    41: "後継者問題と組織内の権力闘争",
    42: "障害の社会モデルと生産性",
    43: "社会変動期における権威と価値観の転換",
    44: "辺境における独自の価値観の形成",
    45: "逆境が促す成長と飛躍",
    46: "軍事的成功体験と過信のリスク",
    47: "強い信念と目標達成の力",
    48: "目標達成のためのリソースと協力体制",
    49: "経済活動におけるコストと価値の交換",
    50: "イノベーションの実践と現実的要因",
    51: "プロパガンダと信頼性の失墜",
    52: "交渉における原則と現実的対応",
    53: "統治の空白と非公式な権力構造",
    54: "過酷な環境が育む強靭な精神性",
    55: "複雑性の理解と多角的視点（メタ認知）",
    56: "非暴力・不服従による抵抗",
    57: "社会秩序と「世間」の圧力",
    58: "内発的動機と自己の探求",
    59: "精神的・経済的支援とコラボレーション",
    60: "執着からの解放と精神的自由",
    61: "メディアによる大衆操作とプロパガンダ",
    62: "長期戦におけるソフトパワーと戦略",
    63: "スケープゴートとしてのマイノリティ",
    64: "貨幣と信用のシステム",
    65: "分断統治と内部対立の助長",
    66: "希少資源をめぐる競争とリーダーシップ",
    67: "信頼の重要性と裏切りの代償",
    68: "ルールと規範の役割",
    69: "危機的状況下における利他行動と共感",
    70: "長期的視点を持つ企業の社会貢献",
    71: "権力者の言動と社会的反発",
    72: "異文化体験と自己の相対化",
    73: "社会構造とルール（ロシア語キーワード）",
    74: "複雑な課題への戦略的アプローチ",
    75: "年齢による社会的役割と標準化",
    76: "社会変化における意識と制度のズレ",
    77: "思想・宗教の普及と実践",
    78: "抽象概念の具現化とデザインの力",
    79: "メディアの選択と情報ニーズ",
    80: "大国間の誤解と意図しない対立激化",
    81: "多様な社会の統治と共通理念",
    82: "豊かさの中の精神的葛藤と自由への希求",
    83: "理想の裏の現実政治（Realpolitik）",
    84: "実務経験を通じた本質的理解",
    85: "リーダーシップと組織的基盤",
    86: "内部結束の欠如と作戦の失敗",
    87: "利害に基づく戦略的同盟関係",
    88: "ネガティブな感情の創造的昇華",
    89: "非人間化のプロセスと共感の欠如",
    90: "属人的支配からシステムによる統治へ",
    91: "無償の行為（利他主義）の価値",
    92: "圧倒的成果による規則違反の正当化",
    93: "社会・文化の基底にある構造",
    94: "征服・統合における文化の強制と反発",
    95: "実力主義と社会階層の流動性",
    96: "直感的・体験的学習の重要性",
    97: "政治的枠組みによる人道的危機の助長",
    98: "システム内での戦略的立ち回り",
    99: "主流思想の空白と代替的価値観の興隆",
    100: "禁止される行為の地下化・制度化",
    101: "対立概念の統合と発展（止揚）",
    102: "欠乏感が煽る情熱と承認欲求",
    103: "社会規範からの逸脱と新たな価値の創造",
    104: "社会問題の認識と公的介入の正当化",
    105: "段階的な社会変革のプロセス",
    106: "組織における構造的なスケープゴート",
    107: "イデオロギーによる現実解釈と意味付け",
    -1: "その他・外れ値"
}

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

def get_topic_display_name(topic_id):
    """Get display name for topic using manual mapping or default format."""
    if topic_id in TOPIC_NAME_MAPPING:
        return f"Topic {topic_id}: {TOPIC_NAME_MAPPING[topic_id]}"
    elif topic_id == -1:
        return BERTOPIC_OUTLIER_LABEL
    else:
        return f"Topic {topic_id}: 未分類"

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
            language="japanese", # Set language for proper tokenization
            verbose=True
        )

        # Fit BERTopic model using original texts and pre-calculated embeddings
        topic_ids, _ = topic_model.fit_transform(original_texts, embeddings=embeddings_array)

        # Convert integer topic IDs to string labels for plotting using manual mapping
        bertopic_labels_str = [get_topic_display_name(label) for label in topic_ids]
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


    # 9. Prepare Data for Plotting (Sort by Topic Name for Legend Order)
    print("Preparing data for plotting (sorting by topic name)...")
    # Combine all data points into a list of tuples
    plot_data = list(zip(
        umap_plot_result[:, 0],
        umap_plot_result[:, 1],
        bertopic_labels_str,
        texts,
        details_texts,
        labels
    ))
    # Sort the data based on the topic ID (extracted from topic name)
    def get_topic_sort_key(topic_name):
        if "Topic -1:" in topic_name or "Outliers" in topic_name:
            return -1  # Outliers first
        elif "Topic " in topic_name:
            # Extract topic number from "Topic X: ..."
            try:
                topic_num = int(topic_name.split(":")[0].replace("Topic ", ""))
                return topic_num
            except ValueError:
                return 9999  # Unknown format goes to end
        else:
            return 9999  # Unknown format goes to end
    
    plot_data_sorted = sorted(plot_data, key=lambda item: get_topic_sort_key(item[2]))

    # Unzip the sorted data back into separate lists/arrays
    sorted_x = [item[0] for item in plot_data_sorted]
    sorted_y = [item[1] for item in plot_data_sorted]
    sorted_topic_labels = [item[2] for item in plot_data_sorted]
    sorted_texts = [item[3] for item in plot_data_sorted]
    sorted_details = [item[4] for item in plot_data_sorted]
    sorted_original_labels = [item[5] for item in plot_data_sorted]

    # 10. Generate Plot with UMAP results and BERTopic labels
    print("Generating plot...")
    try:
        fig = px.scatter(
            x=sorted_x,
            y=sorted_y,
            color=sorted_topic_labels, # Color by sorted BERTopic label
            custom_data=[sorted_texts, sorted_details, sorted_original_labels, sorted_topic_labels], # Use sorted custom data
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
