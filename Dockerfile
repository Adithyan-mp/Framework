
FROM pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime

WORKDIR /test

COPY requirement.txt .

RUN python -m pip install --no-cache-dir --break-system-packages -r requirement.txt

COPY . .

CMD ["python", "-m", "v1_unit_test.main"]
