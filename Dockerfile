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

WORKDIR /app

# Copy only the user-installed packages from builder
COPY --from=builder /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

COPY app/ app/
COPY scripts/ scripts/
COPY .streamlit/ .streamlit/
COPY requirements.txt .

RUN mkdir -p data reports

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", \
     "--server.headless=true", "--server.address=0.0.0.0"]