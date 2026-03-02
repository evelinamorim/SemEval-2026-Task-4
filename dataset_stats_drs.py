"""
Compute DRS statistics from a directory of DRS files.

Each triplet has three files:
  - <idx>_anchor_drs.txt
  - <idx>_a_drs.txt
  - <idx>_b_drs.txt

Usage:
    python dataset_stats_drs.py <drs_dir>
"""

import os
import re
import sys
import statistics
from collections import defaultdict


def parse_drs_file(filepath):
    """Parse a DRS file and return counts of events, participants, temporal relations,
    semantic relations, and objIdentity relations."""

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into sections
    sections = re.split(r'»\s+(\w+)', content)
    # sections will be: ['', 'EVENTS', '...', 'ACTORS', '...', 'RELATIONS', '...']
    section_map = {}
    for i in range(1, len(sections), 2):
        section_map[sections[i].strip().upper()] = sections[i + 1]

    # Count events: lines starting with # T
    event_lines = [l for l in section_map.get("EVENTS", "").splitlines()
                   if re.match(r'#\s+T\d+', l.strip())]
    # Each event has a comment line starting with # T<n> (<text>) -> <var>
    # Count unique event variable lines (the # T lines, not FOL/DRS lines)
    events = len([l for l in event_lines if '->' in l])

    # Count participants (actors)
    actor_lines = [l for l in section_map.get("ACTORS", "").splitlines()
                   if re.match(r'#\s+T\d+\s+->', l.strip())]
    participants = len(actor_lines)

    # Parse relations
    temporal_rels = 0
    semantic_rels = 0
    obj_identity_rels = 0

    temporal_types = {"occursbefore", "occursafter", "overlaps", "during",
                      "completedbefore", "starts", "finishes", "equals", "meets"}


    # Count temporal relations from event DRS lines
    for line in section_map.get("EVENTS", "").splitlines():
        if not line.strip().startswith("# DRS:"):
            continue
        for trel in temporal_types:
            temporal_rels += len(re.findall(rf'\b{trel}\(', line.lower()))

    for line in section_map.get("RELATIONS", "").splitlines():
        m = re.match(r'#\s+T\d+\s+-\s+(\w+)\s+-\s+T\d+', line.strip())
        if not m:
            continue
        rel_type = m.group(1).lower()
        if rel_type == "objidentity":
            obj_identity_rels += 1
        else:
            semantic_rels += 1

    return {
        "events": events,
        "participants": participants,
        "temporal_rels": temporal_rels,
        "semantic_rels": semantic_rels,
        "obj_identity_rels": obj_identity_rels,
    }


def collect_stats(drs_dir):
    """Collect per-role stats from all DRS files in the directory."""

    role_counts = {
        "anchor": defaultdict(list),
        "a": defaultdict(list),
        "b": defaultdict(list),
    }

    # Find all anchor files to determine triplet indices
    all_files = os.listdir(drs_dir)
    anchor_files = [f for f in all_files if f.endswith("_anchor_drs.txt")]

    if not anchor_files:
        print("No anchor DRS files found in directory.")
        sys.exit(1)

    for anchor_file in sorted(anchor_files):
        idx = anchor_file.replace("_anchor_drs.txt", "")
        triplet = {
            "anchor": os.path.join(drs_dir, f"{idx}_anchor_drs.txt"),
            "a":      os.path.join(drs_dir, f"{idx}_a_drs.txt"),
            "b":      os.path.join(drs_dir, f"{idx}_b_drs.txt"),
        }

        for role, path in triplet.items():
            if not os.path.exists(path):
                continue
            try:
                counts = parse_drs_file(path)
                for key, val in counts.items():
                    role_counts[role][key].append(val)
            except Exception as e:
                print(f"Warning: could not parse {path}: {e}")

    return role_counts


def print_stats(role_counts):
    metrics = [
        ("events",          "Events"),
        ("participants",     "Participants"),
        ("temporal_rels",    "Temporal relations"),
        ("semantic_rels",    "Semantic relations"),
        ("obj_identity_rels","objIdentity (coreference)"),
    ]

    role_labels = {"anchor": "Anchor", "a": "Story A", "b": "Story B"}

    # Aggregate across all roles for overall stats
    all_counts = defaultdict(list)
    for role_data in role_counts.values():
        for key, vals in role_data.items():
            all_counts[key].extend(vals)

    print("=" * 60)
    print("DRS DATASET STATISTICS")
    print("=" * 60)

    print("\nOVERALL (all roles combined)")
    print("-" * 60)
    for key, label in metrics:
        vals = all_counts[key]
        avg = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {label}")
        print(f"    Total:   {sum(vals)}")
        print(f"    Average: {avg:.2f} (±{std:.2f})")

    for role, label in role_labels.items():
        print(f"\n{label.upper()}")
        print("-" * 60)
        for key, metric_label in metrics:
            vals = role_counts[role].get(key, [])
            if not vals:
                print(f"  {metric_label}: no data")
                continue
            avg = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"  {metric_label}")
            print(f"    Total:   {sum(vals)}")
            print(f"    Average: {avg:.2f} (±{std:.2f})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dataset_stats_drs.py <drs_dir>")
        sys.exit(1)

    drs_dir = sys.argv[1]
    role_counts = collect_stats(drs_dir)
    print_stats(role_counts)