import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_with_offsets(text):
    return [(match.group(0), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def spans_to_bio(text, entities):
    tokens = tokenize_with_offsets(text)
    tags = ["O"] * len(tokens)
    occupied = set()
    for entity in entities:
        start, end, entity_type = entity.get("start"), entity.get("end"), entity.get("label")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end > len(text):
            raise ValueError("Entity offsets must be valid half-open character spans")
        token_indices = [index for index, (_, token_start, token_end) in enumerate(tokens)
                         if token_start < end and start < token_end]
        if not token_indices:
            raise ValueError("Entity span does not overlap a token")
        first, last = token_indices[0], token_indices[-1]
        if tokens[first][1] != start or tokens[last][2] != end:
            raise ValueError("Entity offsets must align exactly to token boundaries")
        if any(index in occupied for index in token_indices):
            raise ValueError("Overlapping entity spans are not supported by this baseline")
        occupied.update(token_indices)
        tags[first] = "B-" + str(entity_type)
        for index in token_indices[1:]:
            tags[index] = "I-" + str(entity_type)
    return tokens, tags


def token_features(tokens, index):
    token = tokens[index][0]
    previous_token = tokens[index - 1][0] if index > 0 else "<BOS>"
    next_token = tokens[index + 1][0] if index + 1 < len(tokens) else "<EOS>"
    return {
        "token": token,
        "token_lower": token.lower(),
        "prefix2": token[:2],
        "suffix2": token[-2:],
        "prev_lower": previous_token.lower(),
        "next_lower": next_token.lower(),
        "is_digit": token.isdigit(),
        "has_ascii": any(ord(char) < 128 for char in token),
        "has_telugu": any("\u0c00" <= char <= "\u0c7f" for char in token),
    }


def bio_to_spans(tokens, tags):
    spans = []
    active_type = None
    active_start = None
    active_end = None
    for token, tag in zip(tokens, tags):
        prefix, _, entity_type = tag.partition("-")
        if prefix == "B" or (prefix == "I" and active_type != entity_type):
            if active_type is not None:
                spans.append((active_start, active_end, active_type))
            active_type = entity_type
            active_start = token[1]
            active_end = token[2]
        elif prefix == "I" and active_type == entity_type:
            active_end = token[2]
        else:
            if active_type is not None:
                spans.append((active_start, active_end, active_type))
            active_type = None
            active_start = None
            active_end = None
    if active_type is not None:
        spans.append((active_start, active_end, active_type))
    return spans


def entity_metrics(expected_rows, predicted_rows):
    true_entities = set()
    predicted_entities = set()
    for row_index, (expected, predicted) in enumerate(zip(expected_rows, predicted_rows)):
        true_entities.update((row_index, *span) for span in expected)
        predicted_entities.update((row_index, *span) for span in predicted)
    true_positive = len(true_entities & predicted_entities)
    false_positive = len(predicted_entities - true_entities)
    false_negative = len(true_entities - predicted_entities)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    entity_types = sorted({span[2] for span in true_entities | predicted_entities})
    by_type = {}
    for entity_type in entity_types:
        expected_type = {span for span in true_entities if span[3] == entity_type}
        predicted_type = {span for span in predicted_entities if span[3] == entity_type}
        tp = len(expected_type & predicted_type)
        fp = len(predicted_type - expected_type)
        fn = len(expected_type - predicted_type)
        type_precision = tp / (tp + fp) if tp + fp else 0.0
        type_recall = tp / (tp + fn) if tp + fn else 0.0
        type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if type_precision + type_recall else 0.0
        by_type[entity_type] = {"precision": type_precision, "recall": type_recall, "f1": type_f1}
    return {"precision": precision, "recall": recall, "f1": f1, "by_entity_type": by_type}


def train_evaluate(train_rows, test_rows):
    if not train_rows or not test_rows:
        raise ValueError("Training and test NER data must both contain reviewed, span-annotated documents")
    vectorizer = DictVectorizer(sparse=True)
    feature_rows = []
    tag_rows = []
    for row in train_rows:
        tokens, tags = spans_to_bio(row["text"], row.get("entities", []))
        feature_rows.extend(token_features(tokens, index) for index in range(len(tokens)))
        tag_rows.extend(tags)
    if len(set(tag_rows)) < 2:
        raise ValueError("At least two BIO labels are required to train the NER baseline")
    features = vectorizer.fit_transform(feature_rows)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(features, tag_rows)

    expected_spans = []
    predicted_spans = []
    for row in test_rows:
        tokens, expected_tags = spans_to_bio(row["text"], row.get("entities", []))
        test_features = vectorizer.transform([token_features(tokens, index) for index in range(len(tokens))])
        predicted_tags = model.predict(test_features)
        expected_spans.append(bio_to_spans(tokens, expected_tags))
        predicted_spans.append(bio_to_spans(tokens, predicted_tags))
    result = entity_metrics(expected_spans, predicted_spans)
    result["document_count"] = len(test_rows)
    result["model"] = "Token-context DictVectorizer + LogisticRegression; exact typed-span evaluation"
    return result


def load_ner_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                if row.get("task") == "legal_ner" and row.get("text") and row.get("entities"):
                    rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate the Week 5 span-based Legal NER baseline.")
    parser.add_argument("--train", default="data/train/tldc_train.jsonl")
    parser.add_argument("--test", default="data/test/tldc_test.jsonl")
    parser.add_argument("--output", default="results/ner_results.json")
    args = parser.parse_args()
    result = train_evaluate(load_ner_rows(args.train), load_ner_rows(args.test))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
