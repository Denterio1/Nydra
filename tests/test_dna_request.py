"""POST /jobs body validation for the dna goal."""
import pytest
from pydantic import TypeAdapter, ValidationError


def _adapter():
    from src.api_schemas import AnyJobRequest
    return TypeAdapter(AnyJobRequest)


def test_dna_body_validates():
    from src.api_schemas import DnaRequest, JobGoal
    req = _adapter().validate_python({"file_id": "abc", "goal": "dna"})
    assert isinstance(req, DnaRequest)
    assert req.goal == JobGoal.DNA


def test_inspect_body_still_validates():
    from src.api_schemas import InspectRequest
    req = _adapter().validate_python({"file_id": "abc", "goal": "inspect"})
    assert isinstance(req, InspectRequest)


def test_unknown_goal_rejected():
    with pytest.raises(ValidationError):
        _adapter().validate_python({"file_id": "abc", "goal": "nope"})