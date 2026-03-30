import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
from cli.main_cli import app


def init():
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY must be valued in environment")
    if "CHROMADB_PATH" not in os.environ:
        raise ValueError("CHROMADB_PATH must be valued in environment")
    if "DATA_PATH" not in os.environ:
        raise ValueError("DATA_PATH must be valued in environment")


def main():
    init()
    app()



if __name__ == "__main__":
    main()