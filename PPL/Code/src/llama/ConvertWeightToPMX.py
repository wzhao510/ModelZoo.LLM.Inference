import argparse
import macaPMX

from macaPMX.model_zoo.llama.huggingface import write_pmx_model

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
    parser.add_argument(
        "--use_safetensors",
        type=bool,
        default=False,
        help="whether using safetensors for original input file",
    )
    args = parser.parse_args()
    write_pmx_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        use_safetensors=args.use_safetensors
    )

if __name__ == "__main__":
    main()
