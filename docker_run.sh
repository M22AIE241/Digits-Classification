docker build -t flask_app . --no-cache

docker run --mount source=saved_model,destination=/app/saved_model flask_app