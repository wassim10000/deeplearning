FROM python:3.9.18

WORKDIR /app

COPY req.txt .
RUN pip install --no-cache-dir -r req.txt

COPY . .

EXPOSE 10000

CMD ["python", "main.py"] 
