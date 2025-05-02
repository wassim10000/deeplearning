FROM python:3.9.18

WORKDIR /app

COPY req.txt .
RUN pip install --no-cache-dir -r req.txt

COPY . .

CMD ["python", "main.py"] 
