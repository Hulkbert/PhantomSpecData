#!/usr/bin/env python
import os
import re
import glob


def parse_sample_id(filename):
    """
    Extracts the sample ID from a filename using a regex.
    Expected format: ...__<number>__<number>.txt
    Example: "Absorber_Coeff_Baseline_USB4F045381__0__0000.txt"
    """
    match = re.search(r'__(\d+)__\d+\.txt$', filename)
    return int(match.group(1)) if match else None


def main():
    # Adjust the glob pattern to point to your directory containing the text files.
    specFiles = glob.glob('sampleSpecData/Data - Absorption/*.txt')

    if not specFiles:
        print("No files found in the specified directory.")
        return

    print("Debugging Sample IDs from File Names:")
    for file in specFiles:
        basename = os.path.basename(file)
        sample_id = parse_sample_id(basename)
        print(f"{basename} -> {sample_id}")


if __name__ == "__main__":
    main()
