from transformers import pipeline
import yt_dlp
import whisper
import os
import warnings
from collections import Counter

# 경고 무시
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def download_audio_from_youtube(youtube_url, audio_file="temp_audio.mp3"):
    """YouTube 링크에서 오디오 추출"""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": audio_file.replace(".mp3", ""),  # 확장자 제외한 파일명 지정
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "keepvideo": False,  # 원본 영상 파일 유지 안 함
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])


def split_text_by_tokens(text, tokenizer, max_length=512):
    """텍스트를 최대 토큰 길이에 맞게 분할"""
    tokens = tokenizer(text)["input_ids"]
    chunks = [tokens[i : i + max_length] for i in range(0, len(tokens), max_length)]
    return chunks


def analyze_sentiment(text, tokenizer):
    """긴 텍스트를 분할해 감정 분석 후 최종 결과 반환"""
    sentiment_analyzer = pipeline("sentiment-analysis")
    chunks = split_text_by_tokens(text, tokenizer)  # 긴 텍스트 분할
    sentiments = []

    for chunk in chunks:
        decoded_chunk = tokenizer.decode(chunk, skip_special_tokens=True)
        sentiment = sentiment_analyzer(decoded_chunk)[0]["label"]
        sentiments.append(sentiment)

    # 가장 많이 등장한 감정을 최종 감정으로 선택
    most_common_sentiment = Counter(sentiments).most_common(1)[0][0]
    return most_common_sentiment


def process_youtube_video_for_sentiment(youtube_url):
    """YouTube 영상에서 오디오 추출, 텍스트 변환 및 감정 분석"""
    audio_file = "temp_audio.mp3"

    print("Downloading audio...")
    download_audio_from_youtube(youtube_url, audio_file)

    print("Transcribing audio...")
    model = whisper.load_model("base", device="cpu")
    result = model.transcribe(audio_file)
    text = result["text"]
    print(f"Transcribed Text: {text}")

    print("Analyzing sentiment...")
    tokenizer = pipeline("sentiment-analysis").tokenizer
    sentiment = analyze_sentiment(text, tokenizer)
    print(f"Sentiment: {sentiment}")

    if os.path.exists(audio_file):
        os.remove(audio_file)
        print(f"Deleted temporary file: {audio_file}")


# 예제 실행
youtube_url = "https://www.youtube.com/watch?v=6oLB2nh_UuE"
process_youtube_video_for_sentiment(youtube_url)
