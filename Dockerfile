# Folosim Python 3.11 slim
FROM python:3.11-slim

# Setăm directorul de lucru în container
WORKDIR /app

# Copiem requirements.txt și instalăm dependențele
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiem tot codul aplicației
COPY . /app

# Creăm folderul pentru baza de date și dăm permisiuni
RUN mkdir -p /app/db && chmod 777 /app/db

# Expunem portul pe care rulează FastAPI
EXPOSE 8000

# Setăm variabila de mediu pentru calea bazei de date
ENV DATABASE_URL="sqlite+aiosqlite:///app/db/contracts.db"

# Comanda de rulare a aplicației
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
