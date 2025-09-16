import argparse
import time
import requests
import json
import numpy as np
import random
from typing import List, Dict, Tuple

def generate_random_number_string(input_len):
    """Generate random number string of exact length"""
    if input_len <= 0:
        return ""
   
    random_digits = ''.join(random.choice('0123456789') for _ in range(input_len))
    return random_digits

def generate_test_data(model_type: str, model_name: str, input_len: int) -> Dict:
    """Generate exact test data with random number strings"""
    random_text = generate_random_number_string(input_len)
    
    if model_type == "embedding":
        return {
            "model": model_name,
            "input": random_text
        }
    
    elif model_type == "rerank":
        # Generate query and documents with exact lengths
        query = generate_random_number_string(input_len)
        documents = [generate_random_number_string(input_len) for _ in range(5)]  # 5 documents
        return {
            "model": model_name,
            "query": query,
            "documents": documents
        }
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def benchmark(model_type: str, model_name: str, port: int, input_len: int, num_tests: int = 10):
    """
    Test model performance
    
    Args:
        model_type: Model type (embedding or rerank)
        model_name: Model name
        port: Xinference service port
        input_len: Exact input length (characters)
        num_tests: Number of test iterations
    """
    # Set up API endpoint
    if model_type == "embedding":
        url = f"http://localhost:{port}/v1/embeddings"
    elif model_type == "rerank":
        url = f"http://localhost:{port}/v1/rerank"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    headers = {"Content-Type": "application/json"}
    
    # Generate test data with exact length using random numbers
    payload = generate_test_data(model_type, model_name, input_len)
    
    print(f"Test Configuration:")
    print(f"  Model Type: {model_type}")
    print(f"  Model Name: {model_name}")
    print(f"  Port: {port}")
    print(f"  Input Length: {input_len} characters")
    print(f"  Test Iterations: {num_tests}")
    print("-" * 60)
    
    # Warm-up request
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            print(f"Warm-up request failed: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return
        print("Warm-up request successful")
    except Exception as e:
        print(f"Warm-up request exception: {e}")
        return
    
    # Performance testing
    inference_times = []
    token_throughputs = []
    successful_tests = 0
    
    for i in range(num_tests):
        try:
            start_time = time.perf_counter()
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            end_time = time.perf_counter()
            
            if response.status_code == 200:
                inference_time = end_time - start_time
                inference_times.append(inference_time)
                
                # Calculate token throughput if usage info is available
                result = response.json()
                if 'usage' in result:
                    total_tokens = result['usage'].get('total_tokens', 0)
                    if total_tokens > 0 and inference_time > 0:
                        tokens_per_second = total_tokens / inference_time
                        token_throughputs.append(tokens_per_second)
                
                successful_tests += 1
                print(f"Test {i+1}/{num_tests}: {inference_time:.4f}s")
            else:
                print(f"Test {i+1} failed: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"Test {i+1} timed out")
        except Exception as e:
            print(f"Test {i+1} exception: {e}")
    
    if successful_tests > 0:
        # Calculate statistics
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        p95_time = np.percentile(inference_times, 95)
        std_time = np.std(inference_times)
        
        print("-" * 60)
        print("PERFORMANCE RESULTS:")
        print(f"  Successful tests: {successful_tests}/{num_tests}")
        print(f"  Average inference time: {avg_time:.4f}s")
        print(f"  Minimum inference time: {min_time:.4f}s")
        print(f"  Maximum inference time: {max_time:.4f}s")
        print(f"  Throughput: {1/avg_time:.2f} requests/second")
        
        if token_throughputs:
            avg_tokens_per_second = np.mean(token_throughputs)
            print(f"  Average tokens per second: {avg_tokens_per_second:.2f}")
        
    else:
        print("All tests failed")

def main():
    parser = argparse.ArgumentParser(description="Test Xinference model performance")
    parser.add_argument("--model-type", type=str, required=True, 
                       choices=["embedding", "rerank"], help="Model type to test")
    parser.add_argument("--model", type=str, required=True, 
                       help="Model name")
    parser.add_argument("--port", type=int, required=True, help="Xinference service port")
    parser.add_argument("--input-len", type=int, required=True, 
                       help="input length")
    parser.add_argument("--num-tests", type=int, default=10, 
                       help="Number of test iterations, default 10")
    
    args = parser.parse_args()
    
    benchmark(
        args.model_type, 
        args.model, 
        args.port, 
        args.input_len, 
        args.num_tests
    )

if __name__ == "__main__":
    main()