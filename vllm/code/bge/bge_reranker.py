#!/usr/bin/env python3
"""
vLLM Inference Script for BGE-Reranker-v2-M3 Model
Using randomly generated numeric texts for performance testing
"""

import argparse
from vllm import LLM
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
import time
import random
from tqdm import tqdm

def setup_args():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(description='Inference BGE-Reranker-v2-M3 model using vLLM')
    parser.add_argument('--model', type=str, default='BAAI/bge-reranker-v2-m3',
                       help='Model name or path')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.8,
                       help='GPU memory utilization ratio')
    parser.add_argument('--max-model-len', type=int, default=512,
                       help='Maximum model length')
    parser.add_argument('--trust-remote-code', action='store_true', default=True,
                       help='Trust remote code')
    parser.add_argument('--dtype', type=str, default='auto',
                       choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
                       help='Model data type')
    
    # Text generation parameters
    parser.add_argument('--num-queries', type=int, default=50,
                       help='Number of random queries to generate')
    parser.add_argument('--docs-per-query', type=int, default=5,
                       help='Number of documents per query')
    parser.add_argument('--min-length', type=int, default=10,
                       help='Minimum length of each text in characters')
    parser.add_argument('--max-length', type=int, default=100,
                       help='Maximum length of each text in characters')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--profile', action='store_true',
                       help='Enable torch.profiler to capture GPU profiling data')
    

    return parser.parse_args()

def generate_random_numeric_texts(num_texts: int, min_length: int, max_length: int, seed: int = 42) -> List[str]:
    """
    Generate random numeric texts for testing
    
    Args:
        num_texts: Number of texts to generate
        min_length: Minimum text length in characters
        max_length: Maximum text length in characters
        seed: Random seed for reproducibility
        
    Returns:
        List[str]: List of randomly generated numeric texts
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    texts = []
    digits = '0123456789'
    
    for i in range(num_texts):
        # Randomly determine text length
        length = random.randint(min_length, max_length)
        
        # Generate random numeric expression
        text = []
        for j in range(length):
            text.append(random.choice(digits))
        
        text_str = ''.join(text)
        texts.append(text_str)
    
    return texts

def get_llm(args) -> LLM:
    """
    Initializes and returns the LLM model for BGE-Reranker-v2-M3
    
    Args:
        args: Command line arguments
        
    Returns:
        LLM: Initialized vLLM model
    """
    return LLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enforce_eager=True,
        disable_log_stats=True,
    )

def calculate_tokens(prompts: List[str], tokenizer) -> Tuple[int, List[int]]:
    """
    Calculate token counts for prompts
    
    Args:
        prompts: List of prompts
        tokenizer: Tokenizer instance
        
    Returns:
        Tuple[int, List[int]]: Total tokens and tokens per prompt
    """
    total_tokens = 0
    tokens_per_prompt = []
    
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        token_count = len(tokens)
        tokens_per_prompt.append(token_count)
        total_tokens += token_count
    
    return total_tokens, tokens_per_prompt

class BGEReranker:
    def __init__(self, args):
        """Initialize vLLM model for BGE-Reranker"""
        print("Loading BGE-Reranker-v2-M3 model...")
        print(f"Model path: {args.model}")
        print(f"GPU memory utilization: {args.gpu_memory_utilization}")
        print(f"Maximum model length: {args.max_model_len}")
        
        start_time = time.time()
        
        self.llm = get_llm(args)
        
        # Load tokenizer for token counting
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        
        load_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        print(f"Model loaded successfully, time taken: {load_time:.2f} ms")
        
    def score(self, queries: List[str], documents: List[str], args) -> Tuple[np.ndarray, float, float]:
        """
        Score query-document pairs and calculate TPS
        
        Args:
            queries: List of queries
            documents: List of documents
            
        Returns:
            Tuple[np.ndarray, float, float]: Scores, TPS and processing time
        """
        if not queries or not documents:
            print("No queries or documents to process")
            return np.array([]), 0.0, 0.0
        
        # Create query-document pairs
        query_doc_pairs = []
        for query in queries:
            for doc in documents:
                query_doc_pairs.append((query, doc))
        
        print(f"Starting to process {len(query_doc_pairs)} query-document pairs...")
        print(f"Queries: {len(queries)}, Documents: {len(documents)}")

        # Calculate total tokens
        prompts = [f"{query} {doc}" for query, doc in query_doc_pairs]
        total_tokens, tokens_per_prompt = calculate_tokens(prompts, self.tokenizer)
        print(f"Total tokens: {total_tokens}")
        print(f"Average tokens per pair: {total_tokens/len(query_doc_pairs):.1f}")
        
        all_scores = []
        

        # Record start time
        start_time = time.time()
            
        batch_queries = [pair[0] for pair in query_doc_pairs]
        batch_documents = [pair[1] for pair in query_doc_pairs]
        
        try:
            if args.profile:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CUDA,
                        torch.profiler.ProfilerActivity.CPU,
                    ],
                    ) as prof:
                        outputs = self.llm.score(batch_queries, batch_documents)
                prof.export_chrome_trace("bge_reranker_profile.json")
                pertable=prof.key_averages().table(sort_by="cuda_time_total",row_limit=60,max_name_column_width=128)
                print('====================profile===================')
                print(pertable)
                print('====================profile===================')

                with open("bge_reranker_profile.txt", "w") as f:
                    f.write(pertable)
            else:
                outputs = self.llm.score(batch_queries, batch_documents)
            # Extract scores from outputs
            batch_scores = []
            for output in outputs:
                if hasattr(output, 'outputs') and hasattr(output.outputs, 'score'):
                    batch_scores.append(output.outputs.score)
                else:
                    # Fallback: use zero score
                    batch_scores.append(0.0)
                    print(f"Warning: Could not extract score")
            
            all_scores.extend(batch_scores)
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Add zero scores for failed batch
            all_scores.extend([0.0] * len(query_doc_pairs))
            return None, None, None
        
        # Record end time
        end_time = time.time()
        
        # Calculate total processing time
        process_time_ms = (end_time - start_time) * 1000
        
        # Convert to numpy array
        scores = np.array(all_scores)
        
        # Calculate TPS
        tps = total_tokens / (process_time_ms / 1000) if process_time_ms > 0 else 0
        
        print(f"Scoring completed")
        print(f"Total processing time: {process_time_ms:.2f} ms")
        print(f"Tokens per Second (TPS): {tps:.2f}")
        
        return scores, tps, process_time_ms

def main():
    """Main function"""
    args = setup_args()
    
    print("=" * 60)
    print("BGE-Reranker-v2-M3 vLLM Inference Script")
    print("=" * 60)
    print(f"Number of queries: {args.num_queries}")
    print(f"Documents per query: {args.docs_per_query}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    
    # Generate random queries and documents
    queries = generate_random_numeric_texts(
        args.num_queries, 
        args.min_length, 
        args.max_length, 
        args.seed
    )
    
    documents = generate_random_numeric_texts(
        args.num_queries * args.docs_per_query,
        args.min_length, 
        args.max_length, 
        args.seed + 1  # Different seed for documents
    )
    
    print(f"Generated {len(queries)} queries and {len(documents)} documents")
    if not queries or not documents:
        print("No queries or documents generated")
        return
    
    # Initialize model
    try:
        reranker = BGEReranker(args)
    except Exception as e:
        print(f"Model loading failed: {e}")
        return
    
    # Score query-document pairs and calculate TPS
    try:
        scores, tps, process_time_ms = reranker.score(queries, documents,args)
        if scores is None:
            return
    except Exception as e:
        print(f"Scoring failed: {e}")
        return
    
    # Output results
    print("\n" + "=" * 60)
    print("Inference Results Statistics:")
    print("=" * 60)
    
    if len(scores) == 0:
        print("No scores generated")
        return

    print(f"Number of query-document pairs: {len(scores)}")
    print(f"Final TPS: {tps:.2f}")
    print(f"Total inference time: {process_time_ms:.2f} ms")
    
    # Calculate statistics
    print(f"Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    print(f"Average score: {np.mean(scores):.4f}")
    print(f"Score standard deviation: {np.std(scores):.4f}")
    print(f"Median score: {np.median(scores):.4f}")
    
    # Calculate score distribution
    score_bins = np.linspace(np.min(scores), np.max(scores), 6)
    hist, _ = np.histogram(scores, bins=score_bins)
    print("\nScore distribution:")
    for i in range(len(hist)):
        lower = score_bins[i]
        upper = score_bins[i+1]
        count = hist[i]
        percentage = (count / len(scores)) * 100
        print(f"  [{lower:.2f}, {upper:.2f}): {count} pairs ({percentage:.1f}%)")

if __name__ == "__main__":
    main()