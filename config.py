import os

class Config:
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.getenv("SECRET_KEY", "96a56876f636db28072e4a0d92902337dee2a32e327ac3acf5a19897cef56eff")

class ProductionConfig(Config):
    pass
