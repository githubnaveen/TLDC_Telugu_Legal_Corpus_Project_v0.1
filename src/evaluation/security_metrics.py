def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def attack_success_rate(successful_attacks, total_attacks):
    return _ratio(successful_attacks, total_attacks)


def robust_accuracy(correct_adversarial, total_adversarial):
    return _ratio(correct_adversarial, total_adversarial)


def detection_rate(detected_attacks, total_attacks):
    return _ratio(detected_attacks, total_attacks)


def false_positive_rate(false_flags, clean_inputs):
    return _ratio(false_flags, clean_inputs)


def citation_integrity(valid_citations, total_citations):
    return _ratio(valid_citations, total_citations)


def retrieval_degradation(clean_metric, attacked_metric):
    if clean_metric <= 0:
        return None
    return (clean_metric - attacked_metric) / clean_metric


def poisoning_impact(clean_metric, poisoned_metric):
    return retrieval_degradation(clean_metric, poisoned_metric)
