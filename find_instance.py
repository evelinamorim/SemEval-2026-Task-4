"""
Find instance ID by temporal relations pattern
"""

import sys
from drs_parser import parse_drs_file

def find_instance_by_relations(drs_dir, jsonl_path, search_words):
    """
    Find which instance contains specific temporal relation words.

    Args:
        drs_dir: Directory with DRS files
        jsonl_path: Path to jsonl
        search_words: List of words to search for (e.g., ['named', 'stolen'])
    """
    import json

    # Load jsonl to get count
    with open(jsonl_path) as f:
        data = [json.loads(line) for line in f]

    print(f"Searching for relations with words: {search_words}")
    print("="*60)

    for idx in range(len(data)):
        # Parse DRS files
        anchor_path = f"{drs_dir}/{idx}_anchor_drs.txt"
        a_path = f"{drs_dir}/{idx}_a_drs.txt"
        b_path = f"{drs_dir}/{idx}_b_drs.txt"

        try:
            anchor = parse_drs_file(anchor_path)
            a = parse_drs_file(a_path)
            b = parse_drs_file(b_path)

            # Get event words from temporal relations
            def get_relation_words(parser):
                words = []
                var_to_word = {e['variable']: e['text'] for e in parser.events}
                for rel in parser.temporal_relations:
                    source_word = var_to_word.get(rel['source'], '')
                    words.append(source_word)
                return words

            anchor_words = get_relation_words(anchor)
            a_words = get_relation_words(a)
            b_words = get_relation_words(b)

            # Check if any search words match - NOW SHOWS WHERE
            matches = []
            for word in search_words:
                if word in anchor_words:
                    matches.append(f"'{word}' in ANCHOR")
                if word in a_words:
                    matches.append(f"'{word}' in STORY A")
                if word in b_words:
                    matches.append(f"'{word}' in STORY B")

            if matches:
                print(f"\n{'='*60}")
                print(f"FOUND: Instance {idx}")
                print(f"Matches: {', '.join(matches)}")
                print(f"{'='*60}")
                print(f"Anchor relations: {anchor_words[:5]}")
                print(f"Story A relations: {a_words[:5]}")
                print(f"Story B relations: {b_words[:5]}")
                print(f"\nAnchor text: {data[idx]['anchor_text'][:100]}...")
                print(f"Story A text: {data[idx]['text_a'][:100]}...")
                print(f"Story B text: {data[idx]['text_b'][:100]}...")

        except:
            continue

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python find_instance.py <drs_dir> <jsonl_path> [search_word1] [search_word2]")
        print("\nExample:")
        print("  python find_instance.py ./drs_dev dev_track_a.jsonl named stolen")
        sys.exit(1)

    drs_dir = sys.argv[1]
    jsonl_path = sys.argv[2]
    search_words = sys.argv[3:] if len(sys.argv) > 3 else ['named', 'stolen']

    find_instance_by_relations(drs_dir, jsonl_path, search_words)
