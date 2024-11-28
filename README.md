# AI-Sentiment_Classification

docker run -d \
    -e YOUTUBE_API_KEY=your_youtube_api_key \
    -e BRAND_DB_USERNAME=your_rds_username \
    -e BRAND_DB_PASSWORD=your_rds_password \
    -e BRAND_DB_URL=your_rds_endpoint \
    -e BRAND_SERVER_PORT=3306 \
    -e BRAND_DB_DATABASE=your_database_name \
    -p 5001:5001 \
    ai-sentiment-classification
