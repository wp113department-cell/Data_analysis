"""
Uploads: customers attach layout PDFs / photos of cabinet lists; background
workers read them with a vision model and hand the agent a clean item list.

    files.py       validate + store uploads, PDF pages -> images, photo cleanup
    vision.py      read pages/photos with the vision model chain (parallel pages)
    celery_app.py  Celery app (Redis broker + results)
    tasks.py       the background task (progress, file-hash cache)
    jobs.py        submit / check jobs (Celery worker, or a thread if none is running)

Parts adapted from MIT-licensed projects:
    zerox (getomni-ai/zerox)                - PDF->image pipeline, parallel page batching
    text-extract-api (CatchTheTornado)      - Celery task with progress + Redis result cache
"""
