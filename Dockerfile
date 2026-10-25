# Same software versions as the challenge's evaluation machine (PyTorch 2.4.1, CUDA 12.1).
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 1000 runner
COPY --chown=runner:runner . .
USER runner
HEALTHCHECK NONE

# Default: package the submission and check that it loads.
CMD ["python", "tools/build_submission.py", "--out", "/workspace/out/submission.zip", "--check"]
