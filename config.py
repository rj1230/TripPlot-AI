import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load .env from this file's folder, not from wherever Streamlit was launched
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH, override=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# Accepts either spelling so a typo in .env doesn't break it
AVIATION_STACK_API_KEY = os.getenv("AVIATION_STACK_API_KEY") or os.getenv(
    "AVIATIONSTACK_API_KEY"
)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_llm():
    return ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
