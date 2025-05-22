import os

from dotenv import load_dotenv

DOTENV_PATH_ENV = "DOTENV_PATH"


def load_from_dotenv(func):
    def wrapper(*args, **kwargs):
        # Default to .env.local DOTENV_PATH is not set
        load_dotenv(os.getenv(DOTENV_PATH_ENV, ".env.local"))
        return func(*args, **kwargs)

    return wrapper
