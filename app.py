from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from transformers import pipeline
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import yt_dlp
import whisper
import os
import sqlite3
from dotenv import load_dotenv
import uuid

load_dotenv()

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
            sentiment_score REAL NOT NULL,
            shorts_id TEXT NOT NULL,
            thumbnail_id TEXT NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()


# 앱 시작 시 데이터베이스 초기화
init_db()


def get_product_data():
    """백엔드 DB에서 product_id와 상품명 가져오기 (임시 예시)"""
    # MongoDB Atlas 또는 다른 데이터베이스에서 데이터 가져오기
    # 여기서는 예제 데이터를 사용합니다.
    return [
        {"product_id": 1, "product_name": "나이키 에어 포스 1"},
        {"product_id": 2, "product_name": "여성용 스커트"},
    ]


def search_youtube_videos(product_name, max_results=10, page_token=None):
    """YouTube API를 사용해 키워드로 숏츠 검색"""
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

    response = (
        youtube.search()
        .list(
            q=product_name,
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
    try:
        result = whisper_model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        print(f"Transcription error for {audio_file}: {e}")
        return None


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
    product_code,
    youtube_url,
    youtube_thumbnail_url,
    sentiment_label,
    sentiment_score,
    shorts_id,
    thumbnail_id,
):
    """SQLite 데이터베이스에 데이터 저장"""
    conn = sqlite3.connect("shorts.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO shorts (product_code, shorts_id, youtube_url, thumbnail_id, youtube_thumbnail_url, sentiment_label, sentiment_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (
            product_code,
            youtube_url,
            youtube_thumbnail_url,
            sentiment_label,
            sentiment_score,
            shorts_id,
            thumbnail_id,
        ),
    )
    conn.commit()
    conn.close()


def process_video(video, product_code):
    """비디오 처리: 오디오 다운로드, 텍스트 변환, 감정 분석"""
    video_id = video["id"]["videoId"]
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    youtube_thumbnail_url = video["snippet"]["thumbnails"]["high"]["url"]

    # shorts_id 및 thumbnail_id 생성
    shorts_id = str(uuid.uuid4())
    thumbnail_id = str(uuid.uuid4())

    audio_file = f"temp_audio_{video_id}.mp3"

    try:
        download_audio(video_id, audio_file)
        text = transcribe_audio(audio_file)
        if text is None:
            return None  # 텍스트가 없으면 건너뛰기

        sentiment_label, sentiment_score = analyze_sentiment(text)
        print(f"Sentiment: {sentiment_label} (Confidence: {sentiment_score:.2f})")

        save_shorts_to_db(
            product_code,
            youtube_url,
            youtube_thumbnail_url,
            sentiment_label,
            sentiment_score,
            shorts_id,
            thumbnail_id,
        )
        return (
            youtube_url,
            youtube_thumbnail_url,
            sentiment_label,
            sentiment_score,
            shorts_id,
            thumbnail_id,
        )

    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        return None  # 오류 발생 시 None을 반환하여 건너뜁니다.

    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)


@app.route("/api/shorts/search/", methods=["POST"])
def search():
    """특정 상품의 숏츠 검색 및 긍정 영상 반환"""
    # 요청에서 product_id와 product_name 가져오기
    data = request.json
    product_id = data.get("product_id")
    product_name = data.get("product_name")

    if not product_id or not product_name:
        return jsonify({"error": "product_id와 product_name이 필요합니다."}), 400

    # YouTube API 호출
    videos, next_page_token = search_youtube_videos(product_name)

    # 결과 저장 리스트 초기화
    results = []
    processed_video_ids = set()  # 중복 방지용

    while len(results) < 3:  # 결과가 3개에 도달할 때까지 반복
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 각 future와 video를 매핑
            futures = {
                executor.submit(process_video, video, product_id): video
                for video in videos
                if video["id"]["videoId"] not in processed_video_ids
            }

            for future in as_completed(futures):
                video = futures[future]  # 현재 future에 해당하는 video 가져오기
                try:
                    result = future.result()
                    if result:
                        (
                            youtube_url,
                            youtube_thumbnail_url,
                            sentiment_label,
                            sentiment_score,
                            shorts_id,
                            thumbnail_id,
                        ) = result

                        # 감정 분석 결과 확인
                        if sentiment_label == "POSITIVE" and sentiment_score >= 0.5:
                            results.append(
                                {
                                    "shorts_id": shorts_id,
                                    "thumbnail_id": thumbnail_id,
                                    "link": youtube_url,
                                    "thumbnail": youtube_thumbnail_url,
                                    "sentiment_label": sentiment_label,
                                    "sentiment_score": sentiment_score,
                                }
                            )
                        # 처리된 영상 ID 추가
                        processed_video_ids.add(video["id"]["videoId"])
                except Exception as e:
                    print(f"Error processing video {video}: {e}")

                if len(results) >= 3:
                    break

        # 다음 페이지가 있으면 추가 검색
        if len(results) < 3 and next_page_token:
            videos, next_page_token = search_youtube_videos(
                product_name, page_token=next_page_token
            )
        else:
            break

    # 응답 반환
    return jsonify({"product_id": product_id, "positive_videos": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
