# file: config.py

import os
from dotenv import load_dotenv


class AppConfig:
    """
    Loads environment variables from a .env file (or the process environment).
    You can extend or rename these fields as needed.
    """

    def __init__(self, env_file: str = ".env"):
        load_dotenv(dotenv_path=env_file)

        # OpenAI config
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.OPENAI_MODEL_NAME = os.getenv(
            "OPENAI_MODEL_NAME", "text-embedding-ada-002"
        )
        self.OPENAI_PROVIDER = os.getenv(
            "OPENAI_PROVIDER", "openai"
        )  # default to "openai"

        # Snowflake config
        self.SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "")
        self.SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER", "")
        self.SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD", "")
        self.SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "")
        self.SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE", "")
        self.SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "")
        self.DEFAULT_TABLE_NAME = os.getenv("DEFAULT_TABLE_NAME", "default_table")
