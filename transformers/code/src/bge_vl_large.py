import time
import torch
import argparse
from modelscope import AutoModel

def test_bge_vl_performance(model_dir, num_iterations=100, warmup_iter=1, 
                           input_text="a cat", image_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading model from: {model_dir}")
    try:
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        model.set_processor(model_dir)
        model = model.to(device)
        model.eval()
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    

    print("Preparing input data...")
    try:
        image_input = image_path
        print(f"Using image from: {image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    print(f"Starting warm-up ({warmup_iter} iteration)...")
    with torch.no_grad():
        try:
            embeddings = model.encode(
                images=image_input,
                text=input_text
            )
        except Exception as e:
            print(f"Error in warm-up: {e}")
            print("This might be due to missing image file, continuing with performance test...")
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    print(f"Starting performance test ({num_iterations} iterations)...")
    total_time = 0.0
    
    with torch.no_grad():
        for i in range(num_iterations):
            try:
                start_time = time.time()
                
                embeddings = model.encode(
                    images=image_input,
                    text=input_text
                )
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{num_iterations} iterations")
                    
            except Exception as e:
                print(f"Error in iteration {i + 1}: {e}")
                break
    
    if num_iterations > 0 and total_time > 0:
        avg_time_per_iteration = total_time / num_iterations
        
        print("\n" + "="*50)
        print("Performance Test Results:")
        print("="*50)
        print(f"Number of iterations: {num_iterations}")
        print(f"Average time per iteration: {avg_time_per_iteration:.4f} seconds")
        print(f"Total testing time: {total_time:.2f} seconds")
        print("="*50)
            
    else:
        print("Performance test failed: No successful iterations")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BGE-VL perf")
    parser.add_argument("--model-dir", type=str, default="BAAI/BGE-VL-large",
                       help="model path")
    parser.add_argument("--num-iterations", type=int, default=100,
                       help="test rounds")
    parser.add_argument("--input-text", type=str, default="describe this picture",
                       help="input text")
    parser.add_argument("--image-path", type=str, default=None,
                       help="path to image")
    
    args = parser.parse_args()
    
    test_bge_vl_performance(
        model_dir=args.model_dir,
        num_iterations=args.num_iterations,
        input_text=args.input_text,
        image_path=args.image_path
    )