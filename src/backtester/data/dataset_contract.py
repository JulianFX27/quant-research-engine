from dataclasses import dataclass
from typing import List, Dict

@dataclass(frozen=True)
class DatasetContract:
    name: str
    version: str
    
    # Schema
    columns: List[str]
    dtypes: Dict[str, str]
    time_column: str
    
    # Semantics
    timezone: str
    bar_interval_minutes: int
    instrument: str
    
    # Invariants
    required_monotonic_time: bool = True
    allow_gaps: bool = False
    allow_duplicates: bool = False
