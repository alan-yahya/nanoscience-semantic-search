import faiss
import pickle
import numpy as np
from pathlib import Path

def load_reference_dois(doi_file: str) -> set:
    """Load reference DOIs from text file."""
    with open(doi_file, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f}

def clean_faiss_index(faiss_file: str, metadata_file: str, reference_dois: set) -> None:
    """
    Clean a FAISS index by keeping only vectors whose DOIs are in the reference set.
    """
    try:
        # Skip if already cleaned
        if 'cleaned' in faiss_file:
            print(f"\nSkipping cleaned file: {faiss_file}")
            return
            
        # Load FAISS index and metadata
        print(f"\nProcessing: {faiss_file}")
        index = faiss.read_index(faiss_file)
        
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
            
        print(f"Original FAISS index dimensions: {index.ntotal} x {index.d}")
        
        # Create mask for valid DOIs
        valid_mask = np.array([
            any(ref_doi in doi for ref_doi in reference_dois) 
            for doi in metadata['dois']
        ])
        valid_indices = np.where(valid_mask)[0]
        
        print(f"Found {len(valid_indices)} vectors with valid DOIs")
        
        # Create new FAISS index with same parameters
        new_index = faiss.IndexFlatL2(index.d)
        
        # Get vectors for valid indices
        valid_vectors = []
        for idx in valid_indices:
            recons = np.zeros(index.d, dtype=np.float32)
            index.reconstruct(int(idx), recons)
            valid_vectors.append(recons)
        
        # Add vectors to new index
        if valid_vectors:
            valid_vectors = np.array(valid_vectors, dtype=np.float32)
            new_index.add(valid_vectors)
        
        # Filter metadata while preserving structure
        new_metadata = {}
        for key in metadata.keys():
            if isinstance(metadata[key], list):
                new_metadata[key] = [metadata[key][i] for i in valid_indices]
            else:
                new_metadata[key] = metadata[key]
        
        # Save new index and metadata
        output_faiss = faiss_file.replace('.faiss', '_cleaned.faiss')
        output_metadata = metadata_file.replace('.pkl', '_cleaned.pkl')
        
        faiss.write_index(new_index, output_faiss)
        with open(output_metadata, 'wb') as f:
            pickle.dump(new_metadata, f)
        
        # Report statistics
        print(f"\nFAISS index cleaned:")
        print(f"Original entries: {index.ntotal}")
        print(f"Valid entries: {new_index.ntotal}")
        print(f"Removed entries: {index.ntotal - new_index.ntotal}")
        print(f"New files created:")
        print(f"FAISS index: {output_faiss}")
        print(f"Metadata: {output_metadata}")
        
        # Print sample of kept and removed DOIs
        print("\nSample of kept DOIs:")
        for doi in list(set(new_metadata['dois']))[:5]:
            print(f"  {doi}")
        
        print("\nSample of removed DOIs:")
        removed_dois = set(metadata['dois']) - set(new_metadata['dois'])
        for doi in list(removed_dois)[:5]:
            print(f"  {doi}")
        
    except Exception as e:
        print(f"Error processing {faiss_file}: {e}")

def main():
    # Configuration
    reference_doi_file = "openai_reference_dois.txt"
    faiss_file = "recursive_paragraph_embeddings.faiss"  # Adjust filename as needed
    metadata_file = "recursive_paragraph_embeddings_metadata.pkl"  # Adjust filename as needed
    
    # Load reference DOIs
    reference_dois = load_reference_dois(reference_doi_file)
    print(f"Loaded {len(reference_dois)} reference DOIs")
    
    # Clean FAISS index
    clean_faiss_index(faiss_file, metadata_file, reference_dois)

if __name__ == "__main__":
    main()