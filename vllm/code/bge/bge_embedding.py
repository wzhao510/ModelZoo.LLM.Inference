import argparse
from vllm import LLM
import torch
import numpy as np
from typing import List, Tuple
import time
import os
from transformers import AutoTokenizer

def setup_args():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(description='Inference BGE-Large-ZH model using vLLM')
    parser.add_argument('--model', type=str, default='BAAI/bge-large-zh',
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
    parser.add_argument('--max-num-seqs', type=int, default=256,
                       help='Maximum number of sequences to process simultaneously')
    parser.add_argument('--max-num-batched-tokens', type=int, default=2048,
                       help='Maximum number of tokens to process in one batch step')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing (number of texts to read and process per batch)')

    parser.add_argument('--text-file', type=str, required=True,
                       help='Read texts from file, one text per line')
    parser.add_argument('--profile', action='store_true',
                       help='Enable torch.profiler to capture GPU profiling data')
    
    return parser.parse_args()

def load_texts_from_file(file_path: str, batch_size: int = 32) -> List[str]:
    """
    Load texts from file, reading exactly batch_size number of non-empty lines
    
    Args:
        file_path: Path to text file
        batch_size: Number of non-empty lines to read from file
        
    Returns:
        List[str]: List of non-empty texts (exactly batch_size lines)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    texts = []
    print(f"Loading texts from file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(texts) >= batch_size:
                break
            line = line.strip()
            if line:  
                texts.append(line)

    if len(texts) < batch_size:
        print(f"Warning: Texts in file ({len(texts)}) is less than batch_size ({batch_size})")
        print(f"will Fill the {batch_size} rows with the character 'a' ")
        texts.extend(['a'] * (batch_size - len(texts)))
    
    return texts

def calculate_tokens(texts: List[str], tokenizer) -> Tuple[int, List[int]]:
    """
    Calculate token counts for texts
    
    Args:
        texts: List of texts
        tokenizer: Tokenizer instance
        
    Returns:
        Tuple[int, List[int]]: Total tokens and tokens per text
    """
    total_tokens = 0
    tokens_per_text = []
    
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        token_count = len(tokens)
        tokens_per_text.append(token_count)
        total_tokens += token_count
    
    return total_tokens, tokens_per_text

def batch_texts(texts: List[str], batch_size: int) -> List[List[str]]:
    """
    Split texts into batches
    
    Args:
        texts: List of texts to batch
        batch_size: Number of texts per batch
        
    Returns:
        List[List[str]]: List of batches
    """
    batches = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batches.append(batch)
    return batches

class BGEEmbedder:
    def __init__(self, args):
        """Initialize vLLM model"""
        print("Loading BGE-Large-ZH model...")
        print(f"Model path: {args.model}")
        print(f"GPU memory utilization: {args.gpu_memory_utilization}")
        print(f"Maximum model length: {args.max_model_len}")
        print(f"Max num sequences: {args.max_num_seqs}")
        print(f"Max num batched tokens: {args.max_num_batched_tokens}")
        
        start_time = time.time()
        
        self.llm = LLM(
            model=args.model,
            trust_remote_code=args.trust_remote_code,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            enforce_eager=True,
            disable_log_stats=True,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
        )
        
        # Load tokenizer for token counting
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        
        load_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        print(f"Model loaded successfully, time taken: {load_time:.2f} ms")
        
    def embed(self, prompts: List[str], args) -> Tuple[np.ndarray, float, float]:
        """
        Generate text embeddings and calculate TPS
        
        Args:
            prompts: List of texts
            
        Returns:
            Tuple[np.ndarray, float, float]: Embeddings, TPS and processing time
        """
        if not prompts:
            print("No texts to process")
            return np.array([]), 0.0, 0.0
            
        batch_size = len(prompts)
        print(f"Starting to process {batch_size} texts...")
        
        # Calculate total tokens
        total_tokens, tokens_per_text = calculate_tokens(prompts, self.tokenizer)
        print(f"Total tokens: {total_tokens}")
        print(f"Average tokens per text: {total_tokens/batch_size:.1f}")
        
        all_embeddings = []
        total_process_time_ms = 0
        
        # 由于已经按批次大小读取，直接处理整个批次
        print(f"Processing batch of size: {batch_size}")
        
        batch_start_time = time.time()
        
        try:
            if args.profile:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CUDA,
                        torch.profiler.ProfilerActivity.CPU,
                    ],
                    ) as prof:
                        embeddings_output = self.llm.embed(prompts)
                prof.export_chrome_trace("bge_embedding_profile.json")
                pertable=prof.key_averages().table(sort_by="cuda_time_total",row_limit=60,max_name_column_width=128)
                print('====================profile===================')
                print(pertable)
                print('====================profile===================')

                with open("bge_embedding_profile.txt", "w") as f:
                    f.write(pertable)
            else:
                embeddings_output = self.llm.embed(prompts)
            
            # Handle different return types from llm.embed()
            if isinstance(embeddings_output, torch.Tensor):
                batch_embeddings = embeddings_output.cpu().numpy()
            elif isinstance(embeddings_output, list):
                batch_embeddings = np.array(embeddings_output)
            elif hasattr(embeddings_output, 'cpu'):
                batch_embeddings = embeddings_output.cpu().numpy()
            else:
                try:
                    batch_embeddings = np.array(embeddings_output)
                except:
                    # If all else fails, create zero embeddings
                    embedding_dim = 1024
                    batch_embeddings = np.zeros((batch_size, embedding_dim))
            
            all_embeddings.append(batch_embeddings)
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Add zero vectors for failed batch
            embedding_dim = 1024
            zero_embeddings = np.zeros((batch_size, embedding_dim))
            all_embeddings.append(zero_embeddings)
            return None, None, None
        
        batch_end_time = time.time()
        batch_process_time_ms = (batch_end_time - batch_start_time) * 1000
        total_process_time_ms += batch_process_time_ms
        
        print(f"Batch processed in {batch_process_time_ms:.2f} ms")

        # Concatenate all embeddings
        if all_embeddings:
            try:
                embeddings = np.concatenate(all_embeddings, axis=0)
            except ValueError as e:
                print(f"Error concatenating embeddings: {e}")
                print("Creating zero embeddings as fallback")
                embedding_dim = 1024
                embeddings = np.zeros((batch_size, embedding_dim))
        else:
            embeddings = np.array([])
        
        tps = total_tokens / (total_process_time_ms / 1000) if total_process_time_ms > 0 else 0
        
        print(f"Embedding generation completed")
        print(f"Total processing time: {total_process_time_ms:.2f} ms")
        print(f"Tokens per Second (TPS): {tps:.2f}")
        
        return embeddings, tps, total_process_time_ms

def main():
    """Main function"""
    args = setup_args()
    
    print("=" * 60)
    print("BGE-Large-ZH vLLM Inference Script")
    print("=" * 60)
    print(f"Input file: {args.text_file}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max sequences: {args.max_num_seqs}")
    print(f"Max batched tokens: {args.max_num_batched_tokens}")
    print("=" * 60)
    
    # Load texts
    try:
        prompts = load_texts_from_file(args.text_file, args.batch_size)
    except Exception as e:
        print(f"Error loading texts: {e}")
        return
    
    if not prompts:
        print("No valid texts found")
        return
    
    # Initialize model
    try:
        embedder = BGEEmbedder(args)
    except Exception as e:
        print(f"Model loading failed: {e}")
        return
    
    # Generate embeddings and calculate TPS
    try:
        embeddings, tps, process_time_ms = embedder.embed(prompts, args)
        if embeddings is None:
            return
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        return
    
    # Output results
    print("\n" + "=" * 60)
    print("Inference Results Statistics:")
    print("=" * 60)
    
    if len(embeddings) == 0:
        print("No embeddings generated")
        return
        
    print(f"Final TPS: {tps:.2f}")
    print(f"Total inference time: {process_time_ms:.2f} ms")

if __name__ == "__main__":
    main()