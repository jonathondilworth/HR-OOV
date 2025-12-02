# imports
import argparse
import os
import json
import re
import sys


def parse_args():
  parser = argparse.ArgumentParser(
    description="Preprocess JSON and JSONL files for OnT & SNOMED~CT"
  )
  # START: args
  # base dir (usually ./data/ont_dataset)
  parser.add_argument(
    "--base-dir", "-d", required=True,
    help="Root dir for OnT training data"
  )
  # strip-parentheses: default True, allow disabling with --no-strip-parentheses
  strip_parens_arg = parser.add_mutually_exclusive_group()
  strip_parens_arg.add_argument(
    "--strip-parentheses", dest="strip_parentheses",
    action="store_true", help="Remove parenthesised substrings from training text (default: true)"
  )
  strip_parens_arg.add_argument(
    "--no-strip-parentheses", dest="strip_parentheses",
    action="store_false", help="Do NOT remove parenthesised substrings from training text"
  )
  parser.set_defaults(strip_parentheses=True)
  # to-lower-case: default False
  parser.add_argument(
    "--to-lower-case", dest="to_lower_case",
    action="store_true",
    help="convert all training text to lower-case"
  )
  # collapse-whitespace: default False
  parser.add_argument(
    "--collapse-whitespace", dest="collapse_whitespace",
    action="store_true",
    help="collapse multiple whitespaces to one and applies trim"
  )
  # END: args
  return parser.parse_args()


def preprocess_string(io_text: str, opts) -> str:
  """apply parenthesise stripping, lower case and whitespace collapse to io_text"""
  if opts.strip_parentheses:
    io_text = re.sub(r"\([^)]*\)", "", io_text)
  if opts.to_lower_case:
    io_text = io_text.lower()
  if opts.collapse_whitespace:
    io_text = re.sub(r"\s+", " ", io_text).strip()
  return io_text


def preprocess_by_obj_recursion(obj, opts):
  """recursively traverse a (parsed) JSON object, applying preprocessing on each 'str' hit"""
  if isinstance(obj, str):
    return preprocess_string(obj, opts)
  elif isinstance(obj, list):
    return [preprocess_by_obj_recursion(item, opts) for item in obj]
  elif isinstance(obj, dict):
    return {k: preprocess_by_obj_recursion(v, opts) for k, v in obj.items()}
  else:
    return obj


def process_json(path, opts):
  """apply normalisation processing to a single JSON obj from disk, 1. reads, 2. processes, 3. writes"""
  with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
  data = preprocess_by_obj_recursion(data, opts)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write("\n")


def process_jsonl(path, opts):
    """apply norm processing to a single JSONL file from disk"""
    lines_out = []
    with open(path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.rstrip()
        if not line:
          continue
        obj = json.loads(line)
        obj = preprocess_by_obj_recursion(obj, opts)
        lines_out.append(json.dumps(obj, ensure_ascii=False))
    with open(path, "w", encoding="utf-8") as f:
      f.write("\n".join(lines_out) + "\n")


def main():
  """find all files in base_dir with `json` or `jsonl` extensions and normalise value text in line with programme arguments"""
  opts = parse_args()
  skipped = {"classes.json", "relations.json"} # !! don't process with these files !! (they contain IRIs as values)
  for root, dirs, files in os.walk(opts.base_dir):
    for fn in files:
      if fn in skipped:
        continue
      full = os.path.join(root, fn)
      if fn.lower().endswith(".jsonl"):
        process_jsonl(full, opts)
        print(f"Processed JSONL: {full}", file=sys.stderr)
      elif fn.lower().endswith(".json"):
        process_json(full, opts)
        print(f"Processed JSON:  {full}", file=sys.stderr)


if __name__ == "__main__":
    main()
