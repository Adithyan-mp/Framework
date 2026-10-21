from dataclasses import dataclass

@dataclass
class TreeConfig:
    name: str
    root: int
    data:dict[int,list[int]]