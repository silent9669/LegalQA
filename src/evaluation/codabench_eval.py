import numpy as np
import nltk
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

def evaluate_predictions(y_pred: dict, y_true: dict) -> dict[str, float]:
    """
    Evaluates y_pred against y_true using the official CodaBench evaluation rules:
    - Whitespace tokenization (exact reference string .split() and prediction .split())
    - NLTK METEOR score as primary metric
    - ROUGE-L fmeasure without stemming as secondary diagnostic metric
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    clean_preds = {
        str(k): (v['answer'] if isinstance(v, dict) and 'answer' in v else str(v))
        for k, v in y_pred.items()
    }
    clean_true = {
        str(k): (v['answer'] if isinstance(v, dict) and 'answer' in v else str(v))
        for k, v in y_true.items()
    }

    ids = [k for k in clean_true if k in clean_preds]
    if not ids:
        return {"meteor": 0.0, "rouge": 0.0}

    rouge_vals = [
        scorer.score(clean_true[k], clean_preds[k])['rougeL'].fmeasure
        for k in ids
    ]
    meteor_vals = [
        meteor_score([clean_true[k].split()], clean_preds[k].split())
        for k in ids
    ]

    return {
        "meteor": float(np.mean(meteor_vals)),
        "rouge": float(np.mean(rouge_vals))
    }
