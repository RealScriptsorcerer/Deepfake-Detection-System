from typing import Dict, List


class SimpleEnsembleAggregator:
    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.weights = weights or {"video": 0.6, "audio": 0.4}

    def aggregate(self, modality_scores: Dict[str, float]) -> float:
        total_weight = 0.0
        total_score = 0.0
        for modality, score in modality_scores.items():
            w = self.weights.get(modality, 0.0)
            total_weight += w
            total_score += w * float(score)
        if total_weight <= 0.0:
            return 0.0
        return float(min(max(total_score / total_weight, 0.0), 1.0))

