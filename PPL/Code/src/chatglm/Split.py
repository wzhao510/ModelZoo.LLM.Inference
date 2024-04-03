import argparse
import macaPMX.model_zoo.chatglm.modeling.SplitModel as SplitModel

def main():
    parser=argparse.ArgumentParser()

    parser.add_argument("--input_dir", help="Location of pmx weights, which contains model folders")
    parser.add_argument("--num_shards", help="num of shards to split", type=int)
    parser.add_argument("--output_dir", help="Location to write PMX model")

    args = parser.parse_args()
    SplitModel.split_pmx_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        num_shards=args.num_shards
    )

if __name__ == "__main__":
    main()

