import pytest
import requests

API_BASE_URL = "http://127.0.0.1:8000"

@pytest.mark.smoke
def test_als_inference_endpoint():
    try:
        payload = {"user_id": 1, "n_recommendations": 5}
        response = requests.post(
            f"{API_BASE_URL}/recommend/als",
            json=payload,
            timeout=10
        )
        assert response.status_code in [200, 201]
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")