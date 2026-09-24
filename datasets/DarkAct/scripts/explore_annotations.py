import os

# Just this one path needs to be correct -- point it at your annotations folder.
ANNOTATIONS_DIR = r"C:\Users\USER\Desktop\SE499\darkact_datasets\ARIM_v1\annotations"


def read_lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()


def main():
    all_files = sorted(f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".txt"))
    print(f"Found {len(all_files)} annotation files in {ANNOTATIONS_DIR}\n")

    contents = {}
    for fname in all_files:
        lines = read_lines(os.path.join(ANNOTATIONS_DIR, fname))
        contents[fname] = lines
        first_line = lines[0] if lines else "(empty)"
        print(f"  {fname:45s} {len(lines):6d} lines   first line: {first_line}")

    print("\nComparing each '...1_annotations.txt' file to its plain-named twin:")
    found_any_pair = False
    for fname in all_files:
        if fname.endswith("1_annotations.txt"):
            plain_name = fname.replace("1_annotations.txt", "_annotations.txt")
            if plain_name in contents:
                found_any_pair = True
                same = contents[fname] == contents[plain_name]
                verdict = "IDENTICAL" if same else "DIFFERENT"
                print(f"  {plain_name}  vs  {fname}: {verdict} "
                      f"({len(contents[plain_name])} vs {len(contents[fname])} lines)")
    if not found_any_pair:
        print("  (no matching plain-named twin found for any '...1_annotations.txt' file)")


if __name__ == "__main__":
    main()