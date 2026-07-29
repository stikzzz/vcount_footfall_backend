import os
import sys
import modal

# Define Modal App
app = modal.App("vcount-footfall-backend")

# Define Container Image with OpenCV system dependencies & Python ML packages
backend_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxrender1", "libxext6")
    .pip_install(
        "flask",
        "flask-cors",
        "flask-sqlalchemy",
        "pyjwt",
        "opencv-python-headless",
        "ultralytics",
        "torch",
        "torchvision",
        "numpy",
        "python-dotenv",
        "requests"
    )
    .add_local_dir(
        ".",
        remote_path="/root/backend",
        ignore=[
            ".git",
            "__pycache__",
            "*.pyc",
            "my_model/train",
            "*.db"
        ]
    )
)

@app.function(
    image=backend_image,
    cpu=2.0,
    memory=2048,
    timeout=300
)
@modal.wsgi_app()
def flask_app():
    backend_dir = "/root/backend"
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    os.chdir(backend_dir)
    
    from app import app as application
    return application
