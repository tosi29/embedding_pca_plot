from google import genai
from google.genai import types
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
import os
import time

# APIキーを設定
GOOGLE_API_KEY = os.environ.get('GEMINI_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)

# input.txtからテキストを読み込み
with open("input.txt", "r") as f:
    texts = [line.strip() for line in f.readlines()]

# テキストのembeddingを生成
embeddings = []
for text in texts:
    if text: # 空行をスキップ
        result = client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=text,
            config=types.EmbedContentConfig(task_type="CLUSTERING"),
        )
        time.sleep(10) # APIのレート制限を避けるために10秒待機
        embeddings.append(result.embeddings[0].values)

# PCAで2次元に次元削減
if embeddings:
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(np.array(embeddings))

    # plotlyで散布図を作成
    fig = px.scatter(
        pca_result, x=0, y=1,
        text=texts, # hover textに元のテキストを表示
        title='PCA of Text Embeddings'
    )
    fig.update_traces(textposition='top center')
    fig.write_html("pca_plot.html")
    print("pca_plot.html を出力しました")
else:
    print("input.txt に有効なテキストデータがありませんでした。")
