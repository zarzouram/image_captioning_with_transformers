from typing import DefaultDict, List, Dict, Mapping
from collections import Counter

Captions = DefaultDict[str, List[List[str]]]
ImagesAndCaptions = Dict[str, Captions]


class BOW(Counter, Mapping[str, int]):
    pass
