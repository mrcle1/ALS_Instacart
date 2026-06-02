import pytest
import requests

API_BASE_URL = "http://127.0.0.1:8000"

@pytest.mark.smoke
def test_api_health_check():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        assert response.status_code == 200
    except requests.exceptions.ConnectionError:
        pytest.fail("API server is not running")

@pytest.mark.smoke
def test_api_recommendations_als():
    try:
        payload = {"user_id": 1, "n_recommendations": 5}
        response = requests.post(
            f"{API_BASE_URL}/recommend/als",
            json=payload,
            timeout=10
        )
        assert response.status_code in [200, 201]
        data = response.json()
        assert "recommendations" in data or "items" in data
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")