from statistics import mean
from typing import Dict, List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score


class BLUE:

    def __init__(self, ngrams: int = 4) -> None:
        self.smoothing = SmoothingFunction().method3

        self.n = ngrams
        weights = [1 / ngrams if i <= ngrams else 0 for i in range(1, 5)]
        self.weights = tuple(weights)

    def __call__(self, references, hypothesis) -> float:
        score = sentence_bleu(references,
                              hypothesis,
                              weights=self.weights,
                              smoothing_function=self.smoothing)
        return score

    def __repr__(self) -> str:
        return f"bleu{self.n}"


class GLEU:

    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwargs):
        return sentence_gleu(*args, **kwargs)

    def __repr__(self):
        return "gleu"


class METEOR:

    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwargs):
        return meteor_score(*args, **kwargs)

    def __repr__(self):
        return "meteor"


class Metrics:

    def __init__(self) -> None:

        bleu1 = BLUE(ngrams=1)
        bleu2 = BLUE(ngrams=2)
        bleu3 = BLUE(ngrams=3)
        self.bleu4 = BLUE(ngrams=4)

        self.gleu = GLEU()
        meteor = METEOR()  # need nltk.download('omw-1.4')

        self.all = [bleu1, bleu2, bleu3, self.bleu4, self.gleu, meteor]

    def calculate(self,
                  refs: List[List[List[str]]],
                  hypos: List[List[str]],
                  train: bool = False) -> Dict[str, float]:
        if train:
            score_fns = [self.bleu4, self.gleu]
        else:
            score_fns = self.all

        score = {}
        for fn in score_fns:
            score[repr(fn)] = mean(list(map(fn, refs, hypos)))

        return score
