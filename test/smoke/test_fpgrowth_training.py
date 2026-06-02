import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

@pytest.mark.smoke
def test_fpgrowth_model_imports():
    try:
        from src.models.fpgrowth_model import FPGrowthRecommender
        assert FPGrowthRecommender is not None
    except ImportError as e:
        pytest.fail(f"Failed to import FPGrowthRecommender: {e}")

@pytest.mark.smoke
def test_fpgrowth_initialization():
    try:
        from src.models.fpgrowth_model import FPGrowthRecommender
        model = FPGrowthRecommender(
            min_support=0.01,
            min_confidence=0.5,
            min_lift=1.0,
            max_len=3
        )
        assert model is not None
    except Exception as e:
        pytest.fail(f"Failed to initialize FPGrowth model: {e}")