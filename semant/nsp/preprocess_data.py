"""Preprocess dataset into simpler format and filter specific language.

    Input
    -----
        File in JSONL format (originally only periodicals.jsonl and books.jsonl)

    Output
    ------
        File in txt format with each text region separated by \n\n

    Date -- 29.05.2023
    Author -- Martin Kostelnik
"""

import sys
import argparse
import json

from nsp_utils import remove_accents


def parse_arguments():
    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, help="Path to dataset.")
    parser.add_argument("--out", required=True, help="Path to output file.")
    parser.add_argument("--lang", default="cs", help="Language filter.")
    parser.add_argument("--remove-accents", action="store_true", help="If set, removes accents from text.")

    args = parser.parse_args()
    return args


def main(args):
    with open(args.data, "r") as json_f:
        json_list = list(json_f)

    with open(args.out, "w") as out_f:
        for entry in json_list:
            entry_json = json.loads(entry)

            for region in entry_json["regions"]:
                try:
                    lang = region["lang"].lower().strip() if region["lang"] else ""
                    lines = region["lines"]
                except KeyError:
                    continue

                if lang != args.lang:
                    continue

                region_txt = ""
                n_lines = 0

                for line in lines:
                    region_txt += f"{line.strip()}\n"
                    n_lines += 1

                if args.remove_accents:
                    region_txt = remove_accents(region_txt)

                # Save only regions with 2 or more lines of text
                if n_lines >= 2:
                    print(region_txt, file=out_f)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
