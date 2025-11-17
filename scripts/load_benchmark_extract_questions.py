from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any, Dict, List

def parse_args():
    """CLI arguments"""
    parser = argparse.ArgumentParser(
        description="Extract questions from a MIRAGE benchmark file."
    )
    parser.add_argument(
        "--dataset",
        "--benchmark",
        "--input",
        "-i",
        dest="input_path",
        required=True,
        help="Path to (MIRAGE) benchmark.json",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        required=True,
        help="Directory in which the questions.json will be written",
    )
    return parser.parse_args()


def extract_questions(bench_json: Dict[str, Any]):
    """parse questions from benchmark.sjon"""    
    questions: List[Dict[str, Any]] = []
    # loop through constituent datasets
    for dataset_name, dataset_blob in bench_json.items():
        if not isinstance(dataset_blob, dict):
            continue
        # loop through each question for this dataset & extract
        for question_id, question_record in dataset_blob.items():
            question_data: Dict[str, Any] = {
                "source_dataset": dataset_name,
                "question_id": str(question_id),
                "question": question_record.get("question", "").strip(),
            }
            metadata = {}
            for key, value in question_record.items():
                if key not in {"question", "options", "answer"}:
                    metadata[key] = value
            question_data["metadata"] = metadata
            # store extracted question data
            questions.append(question_data)

    return questions


def main() -> None:
    # args
    args = parse_args()
    in_path = Path(args.input_path).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    # out path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark-questions.json"
    
    # READ
    with in_path.open("r", encoding="utf-8") as fp:
        raw_json = json.load(fp)    
    
    # PROCESS
    questions_payload = extract_questions(raw_json)
    
    # WRITE
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(questions_payload, fp, ensure_ascii=False, indent=2)

    print(f"Wrote {len(questions_payload):,} questions to {out_path}")


if __name__ == "__main__":
    main()