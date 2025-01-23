import requests
from datetime import datetime
import os
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", None)
if GITHUB_TOKEN is None:
    raise ValueError("GITHUB_TOKEN is not set")
PR_API_URL = "https://api.github.com/repos/pytorch/pytorch/pulls"
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

response = requests.get(PR_API_URL, headers=headers)
if response.status_code == 200:
    remaining = response.headers.get("X-RateLimit-Remaining")
    reset_time = response.headers.get("X-RateLimit-Reset")
    reset_datetime = datetime.fromtimestamp(int(reset_time))
    print(f"Remaining requests: {remaining}")
    print(f"Rate limit resets at: {reset_datetime}")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
