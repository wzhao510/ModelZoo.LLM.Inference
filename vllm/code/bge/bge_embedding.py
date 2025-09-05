#!/usr/bin/env python3
"""
vLLM Inference Script for BGE-Large-ZH Model
Support reading texts from file with input length control and TPS calculation
"""

import argparse
from vllm import LLM
import torch
import numpy as np
from typing import List, Tuple
import time
import os
from transformers import AutoTokenizer
import torch

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

    # File input parameters
    parser.add_argument('--text-file', type=str, required=True,
                       help='Read texts from file, one text per line')
    parser.add_argument('--input-len', type=int, default=-1,
                       help='Number of texts to read, -1 means read all texts')
    
    parser.add_argument('--profile', action='store_true',
                       help='Enable torch.profiler to capture GPU profiling data')
    
    return parser.parse_args()

def load_texts_from_file(file_path: str, max_chars: int = -1) -> List[str]:
    """
    Load texts from file with total character length limitation
    
    Args:
        file_path: Path to text file
        max_chars: Maximum total characters to read from file, -1 means read all
        
    Returns:
        List[str]: List of texts within the character limit
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    texts = []
    total_chars_read = 0
    print(f"Loading texts from file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_chars != -1 and total_chars_read >= max_chars:
                break
                
            line = line.strip()
            if line:  # Skip empty lines
                # Calculate how many characters we can take from this line
                if max_chars == -1:
                    # No limit, take the whole line
                    texts.append(line)
                    total_chars_read += len(line)
                else:
                    chars_remaining = max_chars - total_chars_read
                    if chars_remaining > 0:
                        if len(line) <= chars_remaining:
                            # Whole line fits within the limit
                            texts.append(line)
                            total_chars_read += len(line)
                        else:
                            # Only take part of the line
                            partial_line = line[:chars_remaining]
                            texts.append(partial_line)
                            total_chars_read += len(partial_line)
                            break  # Reached the character limit
    
    print(f"Loaded {len(texts)} texts from file")
    print(f"text:{texts}")
    print(f"Total characters read: {total_chars_read}")
    if max_chars != -1:
        print(f"Character limit: {max_chars} characters")
    
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

class BGEEmbedder:
    def __init__(self, args):
        """Initialize vLLM model"""
        print("Loading BGE-Large-ZH model...")
        print(f"Model path: {args.model}")
        print(f"GPU memory utilization: {args.gpu_memory_utilization}")
        print(f"Maximum model length: {args.max_model_len}")
        
        start_time = time.time()
        
        self.llm = LLM(
            model=args.model,
            trust_remote_code=args.trust_remote_code,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            enforce_eager=True,
            disable_log_stats=True,
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
            
        print(f"Starting to process {len(prompts)} texts...")
        
        # Calculate total tokens
        total_tokens, tokens_per_text = calculate_tokens(prompts, self.tokenizer)
        print(f"Total tokens: {total_tokens}")
        print(f"Average tokens per text: {total_tokens/len(prompts):.1f}")
        
        all_embeddings = []
         
        # Record start time
        start_time = time.time()
        
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
                # If it's a tensor, convert to numpy
                batch_embeddings = embeddings_output.cpu().numpy()
            elif isinstance(embeddings_output, list):
                # If it's a list, convert to numpy array
                batch_embeddings = np.array(embeddings_output)
            elif hasattr(embeddings_output, 'cpu'):
                # If it has .cpu() method (like torch Tensor)
                batch_embeddings = embeddings_output.cpu().numpy()
            else:
                # Fallback: try to convert to numpy array
                try:
                    batch_embeddings = np.array(embeddings_output)
                except:
                    # If all else fails, create zero embeddings
                    embedding_dim = 1024  # BGE-large-zh typically has 1024-dim embeddings
                    batch_embeddings = np.zeros((len(prompts), embedding_dim))
            
            all_embeddings.append(batch_embeddings)           
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Add zero vectors for failed batch
            embedding_dim = 1024  # BGE-large-zh typically has 1024-dim embeddings
            zero_embeddings = np.zeros((len(prompts), embedding_dim))
            all_embeddings.append(zero_embeddings)
        
        # Record end time
        end_time = time.time()
        
        # Calculate total processing time
        process_time_ms = (end_time - start_time) * 1000
        
        # Concatenate all embeddings
        if all_embeddings:
            try:
                embeddings = np.concatenate(all_embeddings, axis=0)
            except ValueError as e:
                print(f"Error concatenating embeddings: {e}")
                print("Creating zero embeddings as fallback")
                embedding_dim = 1024
                embeddings = np.zeros((len(prompts), embedding_dim))
        else:
            embeddings = np.array([])
        
        # Calculate TPS
        tps = total_tokens / (process_time_ms / 1000) if process_time_ms > 0 else 0
        
        print(f"Embedding generation completed")
        print(f"Total processing time: {process_time_ms:.2f} ms")
        print(f"Tokens per Second (TPS): {tps:.2f}")
        
        return embeddings, tps, process_time_ms

def main():
    """Main function"""
    args = setup_args()
    
    print("=" * 60)
    print("BGE-Large-ZH vLLM Inference Script")
    print("=" * 60)
    print(f"Input file: {args.text_file}")
    print(f"Number of texts to read: {'All' if args.input_len == -1 else args.input_len}")
    print("=" * 60)
    
    # Load texts
    try:
        prompts = load_texts_from_file(args.text_file, args.input_len)
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