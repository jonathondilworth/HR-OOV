import argparse
import json
import os


def load_set_from_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return_set = set()
    for x_string in data:
        return_set.add(x_string.lower())
    return return_set


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute lexical overlap between input file and compare file, write the diff to disk"
    )
    parser.add_argument(
        "--input", "-i",
        dest="file_one",
        required=True,
        help="Path to the first JSON file (list of strings)"
    )
    parser.add_argument(
        "--compare", "-c",
        dest="file_two",
        required=True,
        help="Path to the second JSON file (list of strings)"
    )
    parser.add_argument(
        "--output-diff", "-od",
        dest="output_diff",
        help="Directory in which to write 'strings-diff.json' (contains elems in the input file not in the comparison file)"
    )
    parser.add_argument(
        "--output-inter", "-oi",
        dest="output_inter",
        help="Directory in which to write 'strings-inter.json' (contains elems in both the input file and the comparison file)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set1 = load_set_from_json(args.file_one)
    set2 = load_set_from_json(args.file_two)

    intersection = set1 & set2
    coverage_pct = (len(intersection) / len(set1)) * 100
    formatted = f"{coverage_pct:.2f}".rstrip('0').rstrip('.')
    print(f"coverage: {formatted}%")

    # write set difference
    if args.output_diff:
        diff = sorted(set1 - set2)
        os.makedirs(args.output_diff, exist_ok=True)
        output_path = os.path.join(args.output_diff, "strings-diff.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(diff, f, indent=2)
        print(f"Wrote diff to {output_path}")

    # write set intersection
    if args.output_inter:
        inter = sorted(intersection)
        os.makedirs(args.output_inter, exist_ok=True)
        output_path = os.path.join(args.output_inter, "strings-inter.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(inter, f, indent=2)
        print(f"Wrote diff to {output_path}")


if __name__ == "__main__":
    main()
