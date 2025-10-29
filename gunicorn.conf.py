import os

# Server socket
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '8000')}"

# Worker processes
workers = int(os.getenv("WORKERS", "1"))

# Worker class - use Uvicorn worker for ASGI compatibility
worker_class = "uvicorn.workers.UvicornWorker"

# Timeout - set high for ML model inference
# Model inference can take 2-5 seconds per request
timeout = 120
graceful_timeout = 30

# Keepalive must be HIGHER than load balancer idle timeout
# AWS ALB default: 60s, Azure LB default: 240s
# Set to 75s to be safe for AWS with buffer (60 + 15)
keepalive = 75

# Worker connections - limit concurrent connections per worker
# This helps distribute load more evenly across workers
worker_connections = 1000
max_requests = 0
max_requests_jitter = 0

# Preload app disabled - each worker loads independently
# This ensures proper async behavior with FastAPI
preload_app = False

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"

# Process naming
proc_name = "blip-caption-api"

# Worker lifecycle hooks
def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info(f"Starting Gunicorn with {workers} worker(s)")
    server.log.info(f"Binding to {bind}")
    server.log.info(f"Worker class: {worker_class}")


def post_worker_init(worker):
    """Called just after a worker has been initialized."""
    worker.log.info(f"Worker {worker.pid} initialized")


def worker_exit(server, worker):
    """Called just after a worker has been exited."""
    server.log.info(f"Worker {worker.pid} exited")
