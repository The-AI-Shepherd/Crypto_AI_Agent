from flask import Flask
from database_models import db
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def create_app():
    app = Flask(__name__, instance_relative_config=True)

    # Ensure the instance folder exists
    os.makedirs(app.instance_path, exist_ok=True)

    # Path will resolve to project/instance/portfolio.db
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + app.instance_path + "/crypto_data.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    with app.app_context():
        db.create_all()

    return app
