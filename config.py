import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "ML Model API"
    debug: bool = False
    model_path: str = "models/"
    max_features: int = 1000
    
    class Config:
        env_file = ".env"

settings = Settings()
