# Build stage: install dependencies
FROM python:3.11-slim as builder

WORKDIR /tmp
COPY requirements.txt .

# Upgrade pip/setuptools/wheel FIRST — most of the 34 CVEs Docker Scout
# flagged (pip 24.0, wheel 0.45.x, setuptools) come from outdated versions
# of these packaging tools themselves, not your app code.
RUN pip install --no-cache-dir --user --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user -r requirements.txt

# Runtime stage: minimal final image
FROM python:3.11-slim

# Run as an unprivileged user: a bug in Streamlit or in the app then cannot
# modify the image's system files. uid 1000 matches the usual first user on
# a Linux host, so the ./data and ./reports bind mounts stay writable.
RUN useradd --create-home --uid 1000 app

WORKDIR /app

# Copy only the user-installed packages from builder, into the app user's
# home (/root is not readable by a non-root user).
COPY --from=builder --chown=app:app /root/.local /home/app/.local

ENV PATH=/home/app/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

COPY --chown=app:app app/ app/
COPY --chown=app:app scripts/ scripts/
COPY --chown=app:app .streamlit/ .streamlit/
COPY --chown=app:app requirements.txt .

# Ship the database (committed in the repo, ~40 MB): without it a pulled
# image starts but every question fails with "unable to open database file".
# docker-compose still bind-mounts ./data over it for local development.
COPY --chown=app:app data/railpulse_ai.db data/

# /app itself must be writable: app/db.py writes query_audit.log to the cwd.
RUN mkdir -p reports && chown app:app /app data reports

USER app

EXPOSE 8501

# Streamlit's own liveness endpoint. python:3.11-slim has no curl, so the
# check uses the Python standard library (any exception -> exit code 1).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", \
     "--server.headless=true", "--server.address=0.0.0.0"]
