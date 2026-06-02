import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

@pytest.mark.smoke
def test_als_model_imports():
    try:
        from src.models.als_model import ALSRecommender
        assert ALSRecommender is not None
    except ImportError as e:
        pytest.fail(f"Failed to import ALSRecommender: {e}")

@pytest.mark.smoke
def test_als_training_initialization():
    try:
        from src.models.als_model import ALSRecommender
        model = ALSRecommender(
            factors=10,
            regularization=0.01,
            iterations=2,
            use_gpu=False,
            num_threads=1
        )
        assert model is not None
    except Exception as e:
        pytest.fail(f"Failed to initialize ALS model: {e}")