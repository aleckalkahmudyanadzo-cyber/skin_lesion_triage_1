"""
app/__init__.py
Flask application factory.
"""

from flask import Flask

from config import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure runtime folders exist
    config_class.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    config_class.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config_class.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    from app.database import init_db
    with app.app_context():
        init_db()

    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)

    return app
