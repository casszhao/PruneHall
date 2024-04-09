from typing import Any, Dict, List
from pydantic import BaseModel

class Datapoint(BaseModel):
    """datapoint baseclass for consistency"""

    id: str
    document: str
    target_summary: str
    prompt: str

Dataset = List[Datapoint]

class SummaryResult(BaseModel):
    """output datamodel for evaluating summaries"""

    bertscore: Dict[str, List[float] | str]
    rouge: Dict[str, List[float]]


class HallucinationResult(BaseModel):
    """output datamodel for hallucination checks"""

    summac_zs: List[float]
    summac_conv: List[float]
    harim_plus: List[float]


class FinalResult(BaseModel):
    """result"""

    id: str
    document: str
    generated: str

    # summary results
    bertscore: Dict[str, float]
    rouge: Dict[str, float]

    # hallucination results
    summac_zs: float
    summac_conv: float
    harim_plus: float