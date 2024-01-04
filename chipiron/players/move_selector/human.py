from typing import Literal
from dataclasses import dataclass

Human_Name_Literal = 'Human'

@dataclass
class HumanPlayerArgs:
    type: Literal[Human_Name_Literal]  # for serialization
