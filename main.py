import argparse
#, graph_embedding, gcn_model, ltn_model
from read_data import load_drs_instances
from drs2graph import process_triplets

def main():
    parser = argparse.ArgumentParser(description="DRS Narrative Similarity ")
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["graph_embedding", "gcn", "ltn"],
                        help="strategies")
    parser.add_argument("--drs_data", type=str, required=True, help="path to drs data")

    args = parser.parse_args()
    instances = load_drs_instances(args.drs_data)
    data = process_triplets(instances)
    if args.strategy == "graph_embedding":
        pass


if __name__ == "__main__":
    main()