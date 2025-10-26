import requests
import os
from dotenv import load_dotenv
from typing import Dict, List, Any
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

load_dotenv()
API_KEY = str(os.getenv("WATSONX_API_KEY"))
MODEL   = str(os.getenv("WATSONX_STT_MODEL"))


def get_watsonx_headers() -> Dict[str, str]:
    def _get_iam_token(api_key: str) -> str:
        resp = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    headers = {
        "Authorization": "Bearer " + _get_iam_token(API_KEY),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    
    return headers