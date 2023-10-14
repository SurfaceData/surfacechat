from pydantic import BaseSettings
from typing import List, Optional


class AppSettings(BaseSettings):
    api_keys: Optional[List[str]] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        use_enum_values = True


app_settings = AppSettings()