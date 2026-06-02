import pytest
import requests

API_BASE_URL = "http://127.0.0.1:8000"

@pytest.mark.smoke
def test_api_batch_inference():
    try:
        payload = {
            "algorithm": "als",
            "user_ids": [1, 2, 3],
            "n_recommendations": 5
        }
        response = requests.post(
            f"{API_BASE_URL}/infer/batch",
            json=payload,
            timeout=15
        )
        assert response.status_code in [200, 201]
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")