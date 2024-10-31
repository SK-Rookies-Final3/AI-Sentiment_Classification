from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from transformers import pipeline
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import yt_dlp
import whisper
import os
import sqlite3

app = Flask(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# YouTube API 설정
API_KEY = os.getenv("YOUTUBE_API_KEY")  # 환경 변수에서 API 키 가져오기
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Whisper 모델 로드
whisper_model = whisper.load_model("base", device="cpu")

# 감정 분석 모델 초기화
sentiment_analyzer = pipeline("sentiment-analysis")


def init_db():
    conn = sqlite3.connect("shorts.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS shorts (
            shorts_code INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            product_code INTEGER NOT NULL,
            youtube_url TEXT NOT NULL,
            youtube_thumbnail_url TEXT NOT NULL,
            sentiment_label TEXT NOT NULL,
            sentiment_score REAL NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()


# 앱 시작 시 데이터베이스 초기화
init_db()


def search_youtube_videos(query, max_results=10, page_token=None):
    """YouTube API를 사용해 키워드로 숏츠 검색"""
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

    response = (
        youtube.search()
        .list(
            q=query,
            part="id,snippet",
            maxResults=max_results,
            type="video",
            videoDuration="short",  # 숏츠만 검색
            pageToken=page_token,  # 다음 페이지 토큰
        )
        .execute()
    )

    return response.get("items", []), response.get("nextPageToken")


def download_audio(video_id, audio_file="temp_audio.mp3"):
    """유튜브 영상의 오디오 다운로드"""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": audio_file.replace(".mp3", ""),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "keepvideo": False,
    }
    url = f"https://www.youtube.com/watch?v={video_id}"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def transcribe_audio(audio_file):
    """Whisper로 오디오 텍스트 변환"""
    result = whisper_model.transcribe(audio_file)
    return result["text"]


def split_text_by_tokens(text, tokenizer, max_length=512):
    """텍스트를 최대 토큰 길이에 맞게 분할"""
    tokens = tokenizer(text)["input_ids"]
    chunks = [tokens[i : i + max_length] for i in range(0, len(tokens), max_length)]
    return chunks


def analyze_sentiment(text):
    """긴 텍스트를 분할해 감정 분석 수행 후 결과 통합"""
    tokenizer = sentiment_analyzer.tokenizer
    chunks = split_text_by_tokens(text, tokenizer)
    sentiments = []
    scores = []

    for chunk in chunks:
        decoded_chunk = tokenizer.decode(chunk, skip_special_tokens=True)
        result = sentiment_analyzer(decoded_chunk)[0]
        label = result["label"]
        score = result["score"]
        sentiments.append(label)
        scores.append(score)

    # 가장 많이 등장한 감정을 최종 감정으로 선택
    most_common_sentiment = Counter(sentiments).most_common(1)[0][0]
    average_score = sum(scores) / len(scores) if scores else 0.0
    return most_common_sentiment, average_score


def save_shorts_to_db(
    product_code, youtube_url, youtube_thumbnail_url, sentiment_label, sentiment_score
):
    """SQLite 데이터베이스에 데이터 저장"""
    conn = sqlite3.connect("shorts.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO shorts (product_code, youtube_url, youtube_thumbnail_url, sentiment_label, sentiment_score)
        VALUES (?, ?, ?, ?, ?)
    """,
        (
            product_code,
            youtube_url,
            youtube_thumbnail_url,
            sentiment_label,
            sentiment_score,
        ),
    )
    conn.commit()
    conn.close()


def process_video(video, product_code):
    """비디오 처리: 오디오 다운로드, 텍스트 변환, 감정 분석"""
    video_id = video["id"]["videoId"]
    title = video["snippet"]["title"]
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    youtube_thumbnail_url = video["snippet"]["thumbnails"]["high"]["url"]

    audio_file = f"temp_audio_{video_id}.mp3"

    try:
        download_audio(video_id, audio_file)
        text = transcribe_audio(audio_file)

        sentiment_label, sentiment_score = analyze_sentiment(text)
        print(f"Sentiment: {sentiment_label} (Confidence: {sentiment_score:.2f})")

        save_shorts_to_db(
            product_code,
            youtube_url,
            youtube_thumbnail_url,
            sentiment_label,
            sentiment_score,
        )
        return youtube_url, sentiment_label, sentiment_score, youtube_thumbnail_url

    except Exception as e:
        print(f"Error processing video {video_id}: {e}")

    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)


@app.route("/search", methods=["POST"])
def search():
    """POST 요청으로 숏츠 검색 및 긍정 영상 반환"""
    data = request.json
    query = data.get("query", "")
    product_code = data.get("product_code", 0)
    max_results = data.get("max_results", 10)
    max_positive = data.get("max_positive", 5)

    positive_links = []
    next_page_token = None

    while len(positive_links) < max_positive:
        videos, next_page_token = search_youtube_videos(
            query, max_results, next_page_token
        )
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(process_video, video, product_code) for video in videos
            ]
            for future in as_completed(futures):
                try:
                    (
                        youtube_url,
                        sentiment_label,
                        sentiment_score,
                        youtube_thumbnail_url,
                    ) = future.result()
                    if sentiment_label == "POSITIVE" and sentiment_score >= 0.5:
                        positive_links.append(
                            {"link": youtube_url, "thumbnail": youtube_thumbnail_url}
                        )
                        if len(positive_links) >= max_positive:
                            break
                except Exception as e:
                    print(f"Error processing video: {e}")

        if not next_page_token:
            break

    return jsonify({"positive_videos": positive_links})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
