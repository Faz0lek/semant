import os


GT_FOLDER = r"/home/martin/semant/data/periodicals/periodical-sample/gt"
PRED_FOLDER = r"/home/martin/semant/data/periodicals/periodical-sample/sorted"
RAW_FOLDER = r"/home/martin/semant/data/periodicals/periodical-sample"


def main():
    total_paragraphs = 0
    hits = 0

    for filename in os.listdir(PRED_FOLDER):
        if not filename.endswith(".txt"):
            continue

        if filename not in os.listdir(PRED_FOLDER):
            continue

        with open(os.path.join(PRED_FOLDER, filename)) as f:
            pred_regions = f.read().strip().split("\n\n")
            pred_regions = [region for region in pred_regions if region or region != ""]
            print(f"{filename}\t{len(pred_regions)}")

        with open(os.path.join(GT_FOLDER, filename)) as f:
            gt_regions = f.read().strip().split("\n\n")
            gt_regions = [region for region in gt_regions if region or region != ""]

        assert len(pred_regions) == len(gt_regions)
        total_paragraphs += len(pred_regions)

        for gt_region, pred_region in zip(gt_regions, pred_regions):
            pred_region = pred_region.strip()
            gt_region = gt_region.strip()

            if pred_region == gt_region:
                hits += 1

    print(f"{hits}/{total_paragraphs} ({(hits/total_paragraphs*100):.2f} %)")
    print(len(os.listdir(PRED_FOLDER)))


if __name__ == "__main__":
    main()
