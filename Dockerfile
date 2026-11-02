# Multi-stage, uv-based image shared by both app services (Streamlit UI and the
# FastAPI service). The compose services each override `command`; the default
# CMD here runs the Streamlit UI.

# ----- build stage: resolve deps into a venv --------------------------------
FROM python:3.12-slim AS build

# uv binary from the official distroless image (standard uv-in-Docker pattern).
# Pinned, not :latest — this is the tool that installs every other dependency,
# so an unpinned tag means each build pulls whatever moved that day. uv.lock
# can't compensate: it pins what gets installed, not what does the installing.
# Bump deliberately; ideally to a digest (@sha256:...), which also authenticates
# the image rather than just naming a version.
COPY --from=ghcr.io/astral-sh/uv:0.12.12 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Install deps first (cached until pyproject.toml / uv.lock change), before
# copying source, so code edits don't bust the dependency layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ----- runtime stage --------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Non-root user for runtime.
RUN useradd --create-home --uid 10001 appuser

# The resolved venv from the build stage.
COPY --from=build /app/.venv /app/.venv

# App source. Runtime datasets, databases, and model blobs are excluded by
# .dockerignore and supplied through volumes instead.
COPY app.py ./
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY api ./api
COPY re_source ./re_source

# /app/data must be writable by appuser: the upload page writes the
# DuckDB file (plus DuckDB's `.wal` sidecar) there. COPY leaves everything
# root-owned by default, so without this the page fails with a permission
# error under the non-root user below — and a named volume mounted here
# inherits this ownership, so it stays writable too.
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data

# Put the venv on PATH so `streamlit` / `uvicorn` resolve directly.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8501 8000

# Default entry point: Streamlit UI. The FastAPI service overrides `command`.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
