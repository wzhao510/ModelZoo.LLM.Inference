import argparse

from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
from lmdeploy.vl import load_image


def parse_args():
    parser = argparse.ArgumentParser(
        description='An example using lmdeploy.pipeline')
    parser.add_argument('--model-path',
                        type=str,
                        default='',
                        help='the path of the model')
    parser.add_argument('--tp',
                        type=int,
                        default=1,
                        help='GPU number used in tensor parallelism. Should be 2^n')   
    parser.add_argument('--block-size',
                        type=int,
                        default=16,
                        help='The block size for paging cache. Should be 8, 16, 32')
    parser.add_argument('--cache-max-entry-count',
                        type=float,
                        default=0.8,
                        help='The percentage of free gpu memory occupied by the k/v '
                        'cache, excluding weights ') 
    parser.add_argument('--vl',
                        type=bool,
                        default=False,
                        help='is vl model or not')
    parser.add_argument('--image-url',
                        type=str,
                        default='',
                        help='image url or local path for vl model')
    args = parser.parse_args()
    return args


def main(model_path: str, tp:int, block_size: int, cache_max_entry_count: float, vl: bool = False, image_url: str = None):
    pipe = pipeline(model_path, backend_config=PytorchEngineConfig(tp=tp, device_type="maca", block_size=block_size, cache_max_entry_count=cache_max_entry_count))

    # warm up
    response = pipe("How are you?", top_k=1)
    print(response.text)

    # test multi batch
    question = ["How are you?", "Please introduce Shanghai."]
    response = pipe(question,  top_k=1)
    for idx, r in enumerate(response):
        print(f"batch_{idx}:")
        print(f"Q: {question[idx]}")
        print(f"A: {r.text}")

    if vl:
        image = load_image(image_url)

        # test multi session
        sess = pipe.chat(("please describe this image.", image))
        print("session 1: ", sess)
        sess = pipe.chat('What is the woman doing?', session=sess)
        print("session 2: ", sess)


if __name__ == '__main__': 
    args = parse_args()
    main(model_path=args.model_path, tp=args.tp, block_size=args.block_size, cache_max_entry_count=args.cache_max_entry_count, vl=args.vl, image_url=args.image_url)
    
