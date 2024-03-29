import argparse
import macaPMX

from macaPMX.model_zoo.qwen.huggingface import write_pmx_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of HF weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write PMX model",
    )
    args = parser.parse_args()
    write_pmx_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
    )

if __name__ == "__main__":
    main()

