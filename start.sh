set -e

WORKERS=${WORKERS:-1}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

echo "=========================================="
echo "BLIP Caption API - Starting Server"
echo "=========================================="
echo "Configuration:"
echo "  Workers: $WORKERS"
echo "  Host:    $HOST"
echo "  Port:    $PORT"
echo "=========================================="

if [ "$WORKERS" = "1" ]; then
    echo "Starting with Uvicorn (single worker mode)"
    echo "Configuration: Baseline/Cloud-native (Config A/D)"
    exec uvicorn app:app --host "$HOST" --port "$PORT"
else
    echo "Starting with Gunicorn + Uvicorn workers (multi-worker mode)"
    echo "Workers count: $WORKERS"
    exec gunicorn app:app -c gunicorn.conf.py
fi
