"""
run.py
Local development entry point.
    python run.py
Serves on http://127.0.0.1:5000
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=app.config["DEBUG"])
