from googleapiclient.discovery import build
from transformers import pipeline
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import yt_dlp
import whisper
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# YouTube API 설정
API_KEY = os.getenv("YOUTUBE_API_KEY")  # 환경 변수에서 API 키 가져오기
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Whisper 모델 로드
whisper_model = whisper.load_model("base", device="cpu")

# 감정 분석 모델 초기화
sentiment_analyzer = pipeline("sentiment-analysis")


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


def process_video(video):
    """비디오 처리: 오디오 다운로드, 텍스트 변환, 감정 분석"""
    video_id = video["id"]["videoId"]
    title = video["snippet"]["title"]
    thumbnail_url = video["snippet"]["thumbnails"]["high"]["url"]
    print(f"\n검색된 Shorts: {title}")

    # 오디오 파일 다운로드 및 텍스트 변환
    audio_file = f"temp_audio_{video_id}.mp3"

    try:
        # 오디오 다운로드 및 텍스트 변환
        download_audio(video_id, audio_file)
        text = transcribe_audio(audio_file)

        # 감정 분석 수행
        label, score = analyze_sentiment(text)
        print(f"Sentiment: {label} (Confidence: {score:.2f})")

        return video_id, label, score, thumbnail_url

    except Exception as e:
        print(f"Error processing video {video_id}: {e}")

    finally:
        # 파일이 남아있는지 확인하고 안전하게 삭제
        try:
            if os.path.exists(audio_file):
                os.remove(audio_file)
                print(f"Deleted {audio_file}")
        except Exception as e:
            print(f"Failed to delete {audio_file}: {e}")


def get_positive_video_links(query, max_positive=5, confidence_threshold=0.5):
    """긍정적인 숏츠 링크 및 썸네일 수집"""
    positive_links = []
    next_page_token = None

    while len(positive_links) < max_positive:
        videos, next_page_token = search_youtube_videos(
            query, page_token=next_page_token
        )

        with ThreadPoolExecutor(
            max_workers=5
        ) as executor:  # 병렬 처리: 최대 5개의 작업 실행
            futures = [executor.submit(process_video, video) for video in videos]

            for future in as_completed(futures):
                try:
                    video_id, label, score, thumbnail_url = future.result()

                    if label == "POSITIVE" and score >= confidence_threshold:
                        link = f"https://www.youtube.com/watch?v={video_id}"
                        positive_links.append((link, thumbnail_url))
                        print(f"Positive video found: {link}")

                        if len(positive_links) >= max_positive:
                            return positive_links
                except Exception as e:
                    print(f"Error processing video: {e}")

        if not next_page_token:
            print("더 이상 검색할 페이지가 없습니다.")
            break

    return positive_links


# 사용 예제
product = "나이키 에어 포스 1"
product_review = f'"{product}" 후기'
print(product_review)
positive_videos = get_positive_video_links(product_review)

print("\nTop 5 긍정 Shorts:")
for idx, (link, thumbnail) in enumerate(positive_videos, start=1):
    print(f"{idx}. {link}")
    print(f"   썸네일: {thumbnail}")

if len(positive_videos) < 5:
    print("\n경고 : 5개 미만의 Shorts가 검색되었습니다.")
