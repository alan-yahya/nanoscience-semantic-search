#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#HF chunker
#generated using the "base" conda env 

import glob
import os

http_proxy_url = "http://proxy.alcf.anl.gov:3128"
https_proxy_url = "http://proxy.alcf.anl.gov:3128"
os.environ['http_proxy'] = http_proxy_url
os.environ['https_proxy'] = https_proxy_url

from huggingface_hub import login
#read perm
login(token="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")
#write perm
login(token="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")

# Configuration Variables
MODEL_NAME = "Flamenco43/NanoBERT-V2"
DATASET_NAME = "Flamenco43/cleaned_html_nano_papers_31k"
PROXY_URL = "http://proxy.alcf.anl.gov:3128"

# Embedding and Chunking Configuration
EMBEDDING_DIM = 768
MAX_TOKENS = 512
MIN_TOKENS = 64
OVERLAP_RATIO = 0.1
SAVE_INTERVAL = 100

# File Paths
CHECKPOINT_FILE = "recursive_paragraph_embeddings.faiss"
METADATA_FILE = "recursive_paragraph_metadata.pkl"

import os
import re
import torch
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from typing import List, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

class RecursiveChunker:
    def __init__(self, 
                 tokenizer,
                 max_tokens=MAX_TOKENS, 
                 min_tokens=MIN_TOKENS, 
                 overlap_ratio=OVERLAP_RATIO):
        """
        Recursive text chunking strategy.
        
        Args:
            tokenizer: Tokenizer to use for tokenization
            max_tokens: Maximum number of tokens in a chunk
            min_tokens: Minimum number of tokens in a chunk
            overlap_ratio: Proportion of tokens to overlap between chunks
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_ratio = overlap_ratio
    
    def _detect_chunk_boundaries(self, tokens: List[str]) -> List[int]:
        """
        Detect intelligent chunk boundaries based on various heuristics.
        
        Args:
            tokens: List of tokens to be chunked
        
        Returns:
            List of boundary indices
        """
        boundaries = [0]
        current_length = 0
        
        # Heuristics for chunk boundaries
        boundary_indicators = [
            '.',   # Sentence endings
            ',',   # Commas (for more granular breaks)
            ';',   # Semicolons
            ':',   # Colons
            '\n'   # Explicit line breaks
        ]
        
        for i, token in enumerate(tokens):
            current_length += 1
            
            # Check if we've reached maximum chunk size
            if current_length >= self.max_tokens:
                # Find the nearest boundary indicator
                backward_search = max(
                    [j for j in range(i) 
                     if tokens[j] in boundary_indicators] + [0]
                )
                boundaries.append(backward_search)
                current_length = i - backward_search
        
        boundaries.append(len(tokens))
        return boundaries
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Recursively chunk text into semantically meaningful segments.
        
        Args:
            text: Input text to be chunked
        
        Returns:
            List of text chunks
        """
        # Tokenize the entire text
        tokens = self.tokenizer.tokenize(text)
        
        # If text is shorter than max tokens, return as is
        if len(tokens) <= self.max_tokens:
            return [self.tokenizer.convert_tokens_to_string(tokens)]
        
        # Detect chunk boundaries
        boundaries = self._detect_chunk_boundaries(tokens)
        
        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            # Compute overlap
            overlap_size = int(self.max_tokens * self.overlap_ratio)
            
            # Extract chunk with potential overlap
            chunk_tokens = tokens[
                max(0, start): 
                min(len(tokens), end + overlap_size)
            ]
            
            # Convert tokens back to text
            chunk = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk)
        
        return chunks

class EmbeddingsGenerator:
    def __init__(self, model_name=MODEL_NAME):
        """
        Initialize embedding generation with recursive chunking.
        
        Args:
            model_name: Hugging Face model to use for embeddings
        """
        # Proxy setup (consider moving to environment variables)
        os.environ['http_proxy'] = PROXY_URL
        os.environ['https_proxy'] = PROXY_URL
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Initialize recursive chunker
        self.chunker = RecursiveChunker(self.tokenizer)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Initialize FAISS index and metadata
        self.index = self._load_or_create_index()
        self.metadata = self._load_or_create_metadata()
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create a new one."""
        if os.path.exists(CHECKPOINT_FILE):
            index = faiss.read_index(CHECKPOINT_FILE)
            print(f"Resumed from checkpoint: {CHECKPOINT_FILE}")
        else:
            index = faiss.IndexFlatL2(EMBEDDING_DIM)
            print("Starting from scratch.")
        return index
    
    def _load_or_create_metadata(self):
        """Load existing metadata or create a new dictionary."""
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "rb") as f:
                return pickle.load(f)
        else:
            return {
                'titles': [],
                'dois': [],
                'paragraph_ids': [],
                'chunk_ids': []
            }
    
    def _get_unique_doi_count(self):
        """Return the count of unique DOIs already processed."""
        return len(set(self.metadata['dois']))
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text chunk.
        
        Args:
            text: Text chunk to encode
        
        Returns:
            Numpy array of embedding vector
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=MAX_TOKENS
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use CLS token representation
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding.squeeze(0)
    
    def generate_embeddings(self, dataset):
        """
        Generate embeddings for the entire dataset using recursive chunking.
        
        Args:
            dataset: Hugging Face dataset to process
        """
        # Get the count of unique DOIs already processed
        start_idx = self._get_unique_doi_count()
        
        # Print the starting point
        print(f"Continuing from DOI index: {start_idx} (Unique DOIs already processed)")
        
        for paragraph_id, (paragraph, title, doi) in tqdm(
            enumerate(zip(
                dataset["train"]["paragraphs"], 
                dataset["train"]["titles"], 
                dataset["train"]["dois"]
            )), 
            desc="Processing Paragraphs"
        ):
            # Skip paragraphs that have already been processed based on DOI
            if paragraph_id < start_idx:
                continue
    
            # Use recursive chunking
            chunks = self.chunker.chunk_text(paragraph)
            
            for chunk_id, chunk in enumerate(chunks):
                # Generate embedding for each chunk
                vector = self.encode_text(chunk)
                
                # Add to FAISS index
                self.index.add(np.array([vector], dtype=np.float32))
                
                # Store metadata
                self.metadata['titles'].append(title)
                self.metadata['dois'].append(doi)
                self.metadata['paragraph_ids'].append(paragraph_id)
                self.metadata['chunk_ids'].append(chunk_id)
            
            # Periodic checkpointing
            if paragraph_id > 0 and paragraph_id % SAVE_INTERVAL == 0:
                self._save_checkpoint()
        
        # Final save
        self._save_checkpoint()
    
    def _save_checkpoint(self):
        """Save the current FAISS index and metadata."""
        faiss.write_index(self.index, CHECKPOINT_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(self.metadata, f)
        
        # Get the total number of embeddings and unique DOIs
        total_embeddings = self.index.ntotal
        unique_dois = len(set(self.metadata['dois']))
        
        print(f"Checkpoint saved. Total embeddings: {total_embeddings}, Unique DOIs processed: {unique_dois}")

def main():
    # Load the dataset
    dataset = load_dataset(DATASET_NAME)
    
    # Initialize and run embeddings generator
    generator = EmbeddingsGenerator()
    generator.generate_embeddings(dataset)
    
    # Final report
    print(f"Total embeddings created: {generator.index.ntotal}")
    print(f"Metadata entries: {len(generator.metadata['titles'])}")

if __name__ == "__main__":
    main()


# In[ ]:


#code to investigate NanoBERT FAISS DB

import glob
import os

http_proxy_url = "http://proxy.alcf.anl.gov:3128"
https_proxy_url = "http://proxy.alcf.anl.gov:3128"
os.environ['http_proxy']=http_proxy_url
os.environ['https_proxy']=https_proxy_url

from huggingface_hub import login
#read perm
login(token="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")
#write perm
login(token="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")

print("debug: proxy enabled")

import pandas as pd
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("Flamenco43/cleaned_html_nano_papers_31k")
dataset

print("RSC dataset loaded")

import faiss
import numpy as np
import pickle
import torch
from transformers import AutoTokenizer, AutoModel

# Configuration (should match the original embedding generation script)
MODEL_NAME = "Flamenco43/NanoBERT-V2"
CHECKPOINT_FILE = "recursive_paragraph_embeddings.faiss"
METADATA_FILE = "recursive_paragraph_metadata.pkl"
MAX_TOKENS = 512

class SemanticSearchEngine:
    def __init__(self, model_name=MODEL_NAME):
        """
        Initialize the semantic search engine.
        
        Args:
            model_name: Hugging Face model to use for embeddings
        """
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Load FAISS index and metadata
        self.index = faiss.read_index(CHECKPOINT_FILE)
        with open(METADATA_FILE, "rb") as f:
            self.metadata = pickle.load(f)
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for the query text.
        
        Args:
            query: Text to encode
        
        Returns:
            Numpy array of embedding vector
        """
        inputs = self.tokenizer(
            query, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=MAX_TOKENS
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use CLS token representation
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding.squeeze(0)
    
    def semantic_search(self, query: str, top_k: int = 5) -> list:
        """
        Perform semantic search on the index.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
        
        Returns:
            List of top search results with metadata and match scores
        """
        # Encode query
        query_embedding = self.encode_query(query)
        
        # Perform search
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32), 
            top_k
        )
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Calculate semantic match score (lower distance means higher similarity)
            # Normalize distance to a 0-1 scale where 1 is perfect match, 0 is least similar
            match_score = 1 / (1 + dist)
            
            results.append({
                'title': self.metadata['titles'][idx],
                'doi': self.metadata['dois'][idx],
                'paragraph_id': self.metadata['paragraph_ids'][idx],
                'chunk_id': self.metadata['chunk_ids'][idx],
                'distance': dist,
                'match_score': match_score
            })
        
        return results

def main():
    # Example usage
    search_engine = SemanticSearchEngine()
    
    # Example queries
    queries = [
        "machine learning techniques",
        "neural network architecture",
        "data processing strategies"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = search_engine.semantic_search(query)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Title: {result['title']}")
            print(f"DOI: {result['doi']}")
            print(f"Paragraph ID: {result['paragraph_id']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Semantic Match Score: {result['match_score']:.4f}")
            print(f"Raw Distance: {result['distance']:.4f}")

if __name__ == "__main__":
    main()


# In[38]:


#OpenAI chunker

import glob
import os
import json

http_proxy_url = "http://proxy.alcf.anl.gov:3128"
https_proxy_url = "http://proxy.alcf.anl.gov:3128"
os.environ['http_proxy']=http_proxy_url
os.environ['https_proxy']=https_proxy_url

from huggingface_hub import login
#read perm
login(token="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")
#write perm
login(token="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")

import os
import time
import typing
import numpy as np
import faiss
import pickle
import openai
from openai import OpenAI
from typing import List, Dict, Tuple, Optional

def find_last_processed_position():
    try:
        # Load FAISS metadata
        with open("openai-embeddings_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Debug information about metadata
        print("\nMetadata structure:")
        print(f"Keys in metadata: {list(metadata.keys())}")
        print(f"Number of total vectors: {len(metadata['dois'])}")
        
        # Check if metadata is empty
        if not metadata['dois']:
            print("No DOIs found in metadata")
            return 0  # No DOIs processed, return 0
        
        # Get the unique DOIs from the metadata
        unique_dois = set(metadata['dois'])
        print(f"Number of unique DOIs: {len(unique_dois)}")
        
        return len(unique_dois)  # Return the count of unique DOIs
    
    except FileNotFoundError as e:
        print(f"Error: Could not find metadata file - {e}")
        return 0  # No metadata, start from 0
        
    except Exception as e:
        print(f"Error: {e}")
        print("Exception occurred at:")
        import traceback
        traceback.print_exc()
        return 0  # Default return value in case of errors'
            
class RecursiveParagraphEmbedder:
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 embedding_model: str = "text-embedding-3-large",
                 max_tokens: int = 2000,
                 min_tokens: int = 64,
                 overlap_ratio: float = 0.1):
        """
        Initialize the embeddings generator with configurable parameters.
        """
        # Initialize configuration
        self.config = {
            # API and Model Configuration
            'OPENAI_API_KEY': "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",
            'EMBEDDING_MODEL': embedding_model,
            'EMBEDDING_DIM': 3072,  # Dimension for text-embedding-3-large
            
            # Processing Parameters
            'MAX_TOKENS': max_tokens,
            'MIN_TOKENS': min_tokens,
            'OVERLAP_RATIO': overlap_ratio,
            'BATCH_SIZE': 100,
            'MAX_RETRIES': 3,
            
            # Dataset Configuration
            'DATASET_SPLIT': 'train',
            
            # Output Paths
            'FAISS_INDEX_PATH': 'openai-embeddings.faiss',
            'METADATA_PATH': 'openai-embeddings_metadata.pkl',
            'PROGRESS_FILE': 'openai-embedding_progress.json',
            
            # Processing Control
            'CHECKPOINT_FREQUENCY': 10,
            'VERBOSE': True
        }
        
        # Initialize progress tracking
        self.progress = {
            'total_documents': 0,
            'last_processed_index': -1,
            'total_tokens_processed': 0,
            'errors': [],
            'start_time': time.time(),
            'last_update': time.time()
        }
        
        # Secure API key retrieval
        if not self.config['OPENAI_API_KEY']:
            raise ValueError("OpenAI API key must be provided either as argument or environment variable")
        
        # Client initialization
        self.client = OpenAI(api_key=self.config['OPENAI_API_KEY'])
        
        # Embedding configuration
        self.EMBEDDING_MODEL = embedding_model
        self.EMBEDDING_DIM = 3072  # Dimension for text-embedding-3-large
        
        # Recursive chunking parameters
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_ratio = overlap_ratio
    
    def recursive_chunk_text(self, 
                              text: str, 
                              max_tokens: Optional[int] = None) -> List[str]:
        """
        Recursively chunk text into manageable pieces.
        
        Args:
            text (str): Input text to chunk
            max_tokens (int, optional): Maximum tokens per chunk
        
        Returns:
            List of text chunks
        """
        max_tokens = max_tokens or self.max_tokens
        tokens = text.split()
        
        # If text is short enough, return as is
        if len(tokens) <= max_tokens:
            return [text]
        
        # Calculate overlap size
        overlap_size = int(max_tokens * self.overlap_ratio)
        
        # Chunk the text
        chunks = []
        for start in range(0, len(tokens), max_tokens - overlap_size):
            chunk = ' '.join(tokens[start:start + max_tokens])
            chunks.append(chunk)
        
        return chunks


    def get_embeddings(self, 
                       texts: List[str], 
                       start_index: int = 0,  # Accept start_index as parameter
                       batch_size: int = 20, 
                       max_retries: int = 3, 
                       index: Optional[faiss.IndexFlatL2] = None, 
                       metadata: Optional[Dict] = None,
                       save_path: str = 'openai-embeddings.faiss',
                       metadata_path: str = 'openai-embeddings_metadata.pkl') -> np.ndarray:
        """
        Generate embeddings for texts starting from a specific index.
        Args:
            texts (List[str]): Texts to embed.
            start_index (int): Index to start processing from.
            Other parameters remain unchanged.
        """
        all_embeddings = []

        for i in range(start_index, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            retries = 0

            while retries < max_retries:
                try:
                    response = self.client.embeddings.create(
                        model=self.EMBEDDING_MODEL,
                        input=batch
                    )

                    batch_embeddings = [
                        embedding.embedding for embedding in response.data
                    ]

                    # Add to FAISS index
                    if index is not None:
                        index.add(np.array(batch_embeddings, dtype=np.float32))
                        faiss.write_index(index, save_path)

                    # Save metadata incrementally
                    if metadata is not None:
                        with open(metadata_path, 'wb') as f:
                            pickle.dump(metadata, f)

                    all_embeddings.extend(batch_embeddings)
                    break  # Successful embedding

                except Exception as e:
                    print(f"Embedding error (attempt {retries + 1}): {e}")

                    if "rate limit" in str(e).lower():
                        # Exponential backoff for rate limits
                        time.sleep(10 * (2 ** retries))
                        retries += 1
                    else:
                        # For non-rate limit errors, fill with zero embeddings
                        all_embeddings.extend([np.zeros(self.EMBEDDING_DIM) for _ in range(len(batch))])
                        break

            else:
                # If all retries fail, fill with zero embeddings
                print(f"Failed to embed batch after {max_retries} attempts")
                all_embeddings.extend([np.zeros(self.EMBEDDING_DIM) for _ in range(len(batch))])

            # Small delay between batches
            time.sleep(0.5)

        return np.array(all_embeddings, dtype=np.float32)
        
    def process_dataset(self, dataset, start_index: int = 0):
        """Process dataset and generate embeddings with progress tracking."""
        # Get the dataset split
        dataset_split = dataset
        self.progress['total_documents'] = len(dataset_split)
        
        # Initialize metadata
        metadata = {
            'titles': [],
            'dois': [],
            'paragraph_ids': [],
            'chunk_ids': []
        }
        
        # Try to load existing metadata if it exists
        try:
            if os.path.exists(self.config['METADATA_PATH']):
                with open(self.config['METADATA_PATH'], 'rb') as f:
                    metadata = pickle.load(f)
                print(f"Loaded existing metadata with {len(metadata['dois'])} entries")
        except Exception as e:
            print(f"Could not load existing metadata: {e}")
        
        # Initialize FAISS index
        if os.path.exists(self.config['FAISS_INDEX_PATH']):
            index = faiss.read_index(self.config['FAISS_INDEX_PATH'])
            print(f"Loaded existing index with {index.ntotal} vectors")
        else:
            index = faiss.IndexFlatL2(self.EMBEDDING_DIM)
            print("Created new FAISS index")
        
        # Start from the provided start_index
        for idx in range(start_index, len(dataset_split)):
            try:
                doc = dataset_split[idx]
                text = doc['paragraphs']
                chunks = self.recursive_chunk_text(text)
                
                if chunks:
                    chunk_embeddings = self.get_embeddings(chunks)
                    
                    if len(chunk_embeddings) > 0:
                        index.add(chunk_embeddings)
                        
                        title = eval(doc['titles'])[0] if doc['titles'].startswith('[') else doc['titles']
                        doi = eval(doc['dois'])[0] if doc['dois'].startswith('[') else doc['dois']
                        
                        for chunk_id, _ in enumerate(chunks):
                            metadata['titles'].append(title)
                            metadata['dois'].append(doi)
                            metadata['paragraph_ids'].append(idx)
                            metadata['chunk_ids'].append(chunk_id)
                        
                        faiss.write_index(index, self.config['FAISS_INDEX_PATH'])
                        with open(self.config['METADATA_PATH'], 'wb') as f:
                            pickle.dump(metadata, f)
                
                self._save_progress(idx)
                
                if self.config['VERBOSE'] and idx % 10 == 0:
                    print(f"Processed documents {idx}/{self.progress['total_documents']}")
                    print(f"Number of total vectors: {index.ntotal}")
                    
            except Exception as e:
                self._save_progress(idx, str(e))
                print(f"Error processing document {idx}: {e}")
                if self.config['VERBOSE']:
                    debug_doc = {
                        k: v[:100] + '...' if k == 'paragraphs' else v 
                        for k, v in doc.items()
                    }
                    print(f"Document fields: {list(doc.keys())}")
                    print(f"Field types:", {k: type(v) for k, v in doc.items()})
                continue
        
        return index, metadata

    def _save_progress(self, current_index: int, error: str = None):
        """Save current progress to JSON file."""
        self.progress['last_processed_index'] = current_index
        self.progress['last_update'] = time.time()
        
        if error:
            self.progress['errors'].append({
                'index': current_index,
                'error': str(error),
                'timestamp': time.time()
            })
        
        try:
            with open(self.config['PROGRESS_FILE'], 'w') as f:
                json.dump(self.progress, f, indent=2)
                
            if self.config['VERBOSE'] and current_index % self.config['CHECKPOINT_FREQUENCY'] == 0:
                print("\n")
                print(f"Progress saved at index {current_index}")
        except Exception as e:
            print(f"Error saving progress: {e}")

def main():
    # Load dataset with a smaller subset for testing
    from datasets import load_dataset

    dataset = load_dataset("Flamenco43/cleaned_html_nano_papers_31k", split='train')

    # Initialize embedder
    embedder = RecursiveParagraphEmbedder()

    # Find last processed position (unique DOIs count) and pass it as a parameter
    last_position = find_last_processed_position()
    start_index = last_position + 1 if last_position >= 0 else 0

    # Process dataset
    embeddings, metadata = embedder.process_dataset(dataset, start_index=start_index)
    if embeddings is not None and metadata is not None:
        print(f"Processed {len(embeddings)} embeddings")
    else:
        print("Failed to process embeddings")

if __name__ == "__main__":
    main()


# In[ ]:


#code to investigate OpenAI FAISS DB

import faiss
import pickle
import numpy as np

def inspect_faiss_index(index_path, metadata_path, max_display=10):
    """
    Inspect the contents of a FAISS index and its associated metadata
    """
    # Load FAISS index
    print("Loading FAISS Index...")
    index = faiss.read_index(index_path)
    
    # Load metadata
    print("Loading Metadata...")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Basic index information
    print("\n--- FAISS Index Information ---")
    print(f"Total Embeddings: {index.ntotal}")
    print(f"Embedding Dimension: {index.d}")
    
    # Print sample entries from each metadata list
    print("\n--- Sample Metadata Entries ---")
    for key in ['titles', 'dois', 'paragraph_ids', 'chunk_ids']:
        if key in metadata:
            print(f"\n{key.capitalize()}:")
            # Print first few entries
            for i in range(min(5, len(metadata[key]))):
                print(f"  {i}: {metadata[key][i]}")
    
    # Sample FAISS index values
    print("\n--- Sample FAISS Embedding Vectors ---")
    try:
        # Try to extract a few sample embedding vectors
        for i in range(min(5, index.ntotal)):
            try:
                # Reconstruct the vector
                vector = index.reconstruct(i)
                # Print basic stats about the vector
                print(f"Vector {i}:")
                print(f"  First 5 values: {vector[:5]}")
                print(f"  Vector length: {len(vector)}")
                print(f"  Mean: {np.mean(vector):.4f}")
                print(f"  Std Dev: {np.std(vector):.4f}")
            except Exception as e:
                print(f"Error reconstructing vector {i}: {e}")
    
    except Exception as e:
        print(f"Error analyzing FAISS index: {e}")
    
    # Additional index statistics
    print("\n--- Additional Index Statistics ---")
    try:
        # If it's an IndexFlatL2 or similar
        if hasattr(index, 'metric_type'):
            print(f"Metric Type: {index.metric_type}")
        
        # If possible, get index type
        print(f"Index Type: {type(index)}")
    except Exception as e:
        print(f"Error getting additional statistics: {e}")

def main():
    # Paths to your FAISS and metadata files
    index_path = "openai-embeddings.faiss"
    metadata_path = "openai-embeddings_metadata.pkl"
    
    # Inspect the index
    inspect_faiss_index(index_path, metadata_path)

if __name__ == "__main__":
    main()


# In[3]:


#OpenAI- Get Query Embedding

import glob
import os
import json

http_proxy_url = "http://proxy.alcf.anl.gov:3128"
https_proxy_url = "http://proxy.alcf.anl.gov:3128"
os.environ['http_proxy']=http_proxy_url
os.environ['https_proxy']=https_proxy_url

from huggingface_hub import login
#read perm
login(token="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")
#write perm
login(token="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")

import os
import time
import typing
import numpy as np
import faiss
import pickle
import openai
from openai import OpenAI
from typing import List, Dict, Tuple, Optional

def find_last_processed_position():
    try:
        # Load FAISS metadata
        with open("openai-embeddings_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Debug information about metadata
        print("\nMetadata structure:")
        print(f"Keys in metadata: {list(metadata.keys())}")
        print(f"Number of total vectors: {len(metadata['dois'])}")
        
        # Check if metadata is empty
        if not metadata['dois']:
            print("No DOIs found in metadata")
            return 0  # No DOIs processed, return 0
        
        # Get the unique DOIs from the metadata
        unique_dois = set(metadata['dois'])
        print(f"Number of unique DOIs: {len(unique_dois)}")
        
        return len(unique_dois)  # Return the count of unique DOIs
    
    except FileNotFoundError as e:
        print(f"Error: Could not find metadata file - {e}")
        return 0  # No metadata, start from 0
        
    except Exception as e:
        print(f"Error: {e}")
        print("Exception occurred at:")
        import traceback
        traceback.print_exc()
        return 0  # Default return value in case of errors'
            
class RecursiveParagraphEmbedder:
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 embedding_model: str = "text-embedding-3-large",
                 max_tokens: int = 2000,
                 min_tokens: int = 64,
                 overlap_ratio: float = 0.1):
        """
        Initialize the embeddings generator with configurable parameters.
        """
        # Initialize configuration
        self.config = {
            # API and Model Configuration
            'OPENAI_API_KEY': "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",
            'EMBEDDING_MODEL': embedding_model,
            'EMBEDDING_DIM': 3072,  # Dimension for text-embedding-3-large
            
            # Processing Parameters
            'MAX_TOKENS': max_tokens,
            'MIN_TOKENS': min_tokens,
            'OVERLAP_RATIO': overlap_ratio,
            'BATCH_SIZE': 100,
            'MAX_RETRIES': 3,
            
            # Dataset Configuration
            'DATASET_SPLIT': 'train',
            
            # Output Paths
            'FAISS_INDEX_PATH': 'openai-embeddings.faiss',
            'METADATA_PATH': 'openai-embeddings_metadata.pkl',
            'PROGRESS_FILE': 'openai-embedding_progress.json',
            
            # Processing Control
            'CHECKPOINT_FREQUENCY': 10,
            'VERBOSE': True
        }
        
        # Initialize progress tracking
        self.progress = {
            'total_documents': 0,
            'last_processed_index': -1,
            'total_tokens_processed': 0,
            'errors': [],
            'start_time': time.time(),
            'last_update': time.time()
        }
        
        # Secure API key retrieval
        if not self.config['OPENAI_API_KEY']:
            raise ValueError("OpenAI API key must be provided either as argument or environment variable")
        
        # Client initialization
        self.client = OpenAI(api_key=self.config['OPENAI_API_KEY'])
        
        # Embedding configuration
        self.EMBEDDING_MODEL = embedding_model
        self.EMBEDDING_DIM = 3072  # Dimension for text-embedding-3-large
        
        # Recursive chunking parameters
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_ratio = overlap_ratio
    
    def recursive_chunk_text(self, 
                              text: str, 
                              max_tokens: Optional[int] = None) -> List[str]:
        """
        Recursively chunk text into manageable pieces.
        
        Args:
            text (str): Input text to chunk
            max_tokens (int, optional): Maximum tokens per chunk
        
        Returns:
            List of text chunks
        """
        max_tokens = max_tokens or self.max_tokens
        tokens = text.split()
        
        # If text is short enough, return as is
        if len(tokens) <= max_tokens:
            return [text]
        
        # Calculate overlap size
        overlap_size = int(max_tokens * self.overlap_ratio)
        
        # Chunk the text
        chunks = []
        for start in range(0, len(tokens), max_tokens - overlap_size):
            chunk = ' '.join(tokens[start:start + max_tokens])
            chunks.append(chunk)
        
        return chunks


    def get_embeddings(self, 
                   text: str,  # Changed from List[str] to str
                   max_retries: int = 3) -> np.ndarray:  # Simplified parameters
        """
        Generate embedding for a single text query.
        Args:
            text (str): Text to embed
            max_retries (int): Maximum number of retry attempts
        Returns:
            np.ndarray: Embedding vector
        """
        retries = 0

        while retries < max_retries:
            try:
                response = self.client.embeddings.create(
                    model=self.EMBEDDING_MODEL,
                    input=text
                )

                embedding = response.data[0].embedding
                return np.array(embedding, dtype=np.float32)

            except Exception as e:
                print(f"Embedding error (attempt {retries + 1}): {e}")

                if "rate limit" in str(e).lower():
                    # Exponential backoff for rate limits
                    time.sleep(10 * (2 ** retries))
                    retries += 1
                else:
                    # For non-rate limit errors, return zero embedding
                    return np.zeros(self.EMBEDDING_DIM)

        # If all retries fail, return zero embedding
        print(f"Failed to embed text after {max_retries} attempts")
        return np.zeros(self.EMBEDDING_DIM)
        
    def process_dataset(self, dataset, start_index: int = 0):
        """Process dataset and generate embeddings with progress tracking."""
        # Get the dataset split
        dataset_split = dataset
        self.progress['total_documents'] = len(dataset_split)
        
        # Initialize metadata
        metadata = {
            'titles': [],
            'dois': [],
            'paragraph_ids': [],
            'chunk_ids': []
        }
        
        # Try to load existing metadata if it exists
        try:
            if os.path.exists(self.config['METADATA_PATH']):
                with open(self.config['METADATA_PATH'], 'rb') as f:
                    metadata = pickle.load(f)
                print(f"Loaded existing metadata with {len(metadata['dois'])} entries")
        except Exception as e:
            print(f"Could not load existing metadata: {e}")
        
        # Initialize FAISS index
        if os.path.exists(self.config['FAISS_INDEX_PATH']):
            index = faiss.read_index(self.config['FAISS_INDEX_PATH'])
            print(f"Loaded existing index with {index.ntotal} vectors")
        else:
            index = faiss.IndexFlatL2(self.EMBEDDING_DIM)
            print("Created new FAISS index")
        
        # Start from the provided start_index
        for idx in range(start_index, len(dataset_split)):
            try:
                doc = dataset_split[idx]
                text = doc['paragraphs']
                chunks = self.recursive_chunk_text(text)
                
                if chunks:
                    chunk_embeddings = self.get_embeddings(chunks)
                    
                    if len(chunk_embeddings) > 0:
                        index.add(chunk_embeddings)
                        
                        title = eval(doc['titles'])[0] if doc['titles'].startswith('[') else doc['titles']
                        doi = eval(doc['dois'])[0] if doc['dois'].startswith('[') else doc['dois']
                        
                        for chunk_id, _ in enumerate(chunks):
                            metadata['titles'].append(title)
                            metadata['dois'].append(doi)
                            metadata['paragraph_ids'].append(idx)
                            metadata['chunk_ids'].append(chunk_id)
                        
                        faiss.write_index(index, self.config['FAISS_INDEX_PATH'])
                        with open(self.config['METADATA_PATH'], 'wb') as f:
                            pickle.dump(metadata, f)
                
                self._save_progress(idx)
                
                if self.config['VERBOSE'] and idx % 10 == 0:
                    print(f"Processed documents {idx}/{self.progress['total_documents']}")
                    print(f"Number of total vectors: {index.ntotal}")
                    
            except Exception as e:
                self._save_progress(idx, str(e))
                print(f"Error processing document {idx}: {e}")
                if self.config['VERBOSE']:
                    debug_doc = {
                        k: v[:100] + '...' if k == 'paragraphs' else v 
                        for k, v in doc.items()
                    }
                    print(f"Document fields: {list(doc.keys())}")
                    print(f"Field types:", {k: type(v) for k, v in doc.items()})
                continue
        
        return index, metadata

    def _save_progress(self, current_index: int, error: str = None):
        """Save current progress to JSON file."""
        self.progress['last_processed_index'] = current_index
        self.progress['last_update'] = time.time()
        
        if error:
            self.progress['errors'].append({
                'index': current_index,
                'error': str(error),
                'timestamp': time.time()
            })
        
        try:
            with open(self.config['PROGRESS_FILE'], 'w') as f:
                json.dump(self.progress, f, indent=2)
                
            if self.config['VERBOSE'] and current_index % self.config['CHECKPOINT_FREQUENCY'] == 0:
                print("\n")
                print(f"Progress saved at index {current_index}")
        except Exception as e:
            print(f"Error saving progress: {e}")

def main():
    # Initialize embedder
    embedder = RecursiveParagraphEmbedder()
    
    # Get user input
    text = input("Enter the text to embed: ")
    
    # Generate embedding
    embedding = embedder.get_embeddings(text)
    print(f"\nEmbedding shape: {embedding.shape}")
    print(f"First few values: {embedding[:5]}")
    
    # Save embedding to file
    output_file = "openai_query_embedding.npy"
    np.save(output_file, embedding)
    print(f"\nEmbedding saved to {output_file}")

if __name__ == "__main__":
    main()


# In[ ]:


#OpenAI query/retrieval:

import numpy as np
import faiss
import pickle

def calculate_semantic_match_scores(query_embedding_path='openai_query_embedding.npy', 
                                    faiss_index_path='openai-embeddings.faiss', 
                                    metadata_path='openai-embeddings_metadata.pkl',
                                    top_k=10):
    """
    Calculate semantic match scores for a query embedding against the existing document embeddings.
    
    Args:
        query_embedding_path (str): Path to the numpy file containing the query embedding
        faiss_index_path (str): Path to the FAISS index file
        metadata_path (str): Path to the metadata pickle file
        top_k (int): Number of top matches to return
    
    Returns:
        dict: A dictionary containing match scores and metadata for top matches
    """
    # Load query embedding
    try:
        query_embedding = np.load(query_embedding_path)
        
        # Ensure query embedding is 2D (FAISS requires 2D array)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
    except Exception as e:
        print(f"Error loading query embedding: {e}")
        return None
    
    # Load FAISS index
    try:
        index = faiss.read_index(faiss_index_path)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None
    
    # Load metadata
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None
    
    # Perform semantic search
    try:
        # Search for top_k nearest neighbors
        D, I = index.search(query_embedding, top_k)
        
        # Prepare results
        results = {
            'distances': D[0].tolist(),
            'indices': I[0].tolist(),
            'matches': []
        }
        
        # Populate match details
        for dist, idx in zip(D[0], I[0]):
            match_info = {
                'distance': float(dist),  # Convert to standard float
                'similarity_score': 1 / (1 + dist),  # Convert distance to a similarity score
                'title': metadata['titles'][idx],
                'doi': metadata['dois'][idx],
                'paragraph_id': metadata['paragraph_ids'][idx],
                'chunk_id': metadata['chunk_ids'][idx]
            }
            results['matches'].append(match_info)
        
        return results
    
    except Exception as e:
        print(f"Error during semantic search: {e}")
        return None

def main():
    # Run the semantic matching
    results = calculate_semantic_match_scores()
    
    if results:
        print("\nTop Semantic Matches:")
        for i, match in enumerate(results['matches'], 1):
            print(f"\nMatch {i}:")
            print(f"Title: {match['title']}")
            print(f"DOI: {match['doi']}")
            print(f"Distance: {match['distance']:.4f}")
            print(f"Similarity Score: {match['similarity_score']:.4f}")
            print(f"Paragraph ID: {match['paragraph_id']}")
            print(f"Chunk ID: {match['chunk_id']}")

if __name__ == "__main__":
    main()


# In[ ]:


"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"

