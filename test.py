from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
import requests

resp = requests.get(
    "http://api.aviationstack.com/v1/flights",
    params={"access_key": os.getenv("AVIATIONSTACK_API_KEY"), "limit": 1},
)
print(resp.status_code, resp.json())
