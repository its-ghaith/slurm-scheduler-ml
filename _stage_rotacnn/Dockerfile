# syntax=docker/dockerfile:1.4
FROM python:3.12-slim AS build

RUN apt-get update

RUN apt-get install -y --no-install-recommends build-essential

RUN pip install uv

RUN rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only poetry files first (for better caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --group cpu --no-group gpu --no-install-project

# Copy and extract the data subset archive
COPY data_subset.tar.gz /tmp/
RUN mkdir -p /app/data && \
	tar xzf /tmp/data_subset.tar.gz -C /app/ && \
	rm /tmp/data_subset.tar.gz && \
	ls -la /app/data/

# Final runtime image
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install runtime dependencies for OpenCV
RUN apt-get update

RUN apt-get install -y --no-install-recommends \
	libgl1 \
	libglib2.0-0

RUN rm -rf /var/lib/apt/lists/*

# Copy Python packages from build stage
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /app/data /app/data
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV TORCH_COMPILE_DISABLE=1
