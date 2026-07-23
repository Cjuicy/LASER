import torch


def prepare_caches_for_aggregation(raw_predictions, overlap):
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    parsed_caches = []
    for index, prediction in enumerate(raw_predictions):
        trim = overlap if index > 0 else 0
        parsed_caches.append({
            key: value[trim:]
            if isinstance(value, torch.Tensor)
            else value
            for key, value in prediction.items()
        })
    return parsed_caches
