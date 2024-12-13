import pickle

def save_openai_dois(metadata_file: str, output_file: str) -> None:
    """
    Extract and save DOIs from OpenAI metadata file.
    
    Args:
        metadata_file: Path to OpenAI metadata pickle file
        output_file: Path to output text file
    """
    try:
        # Load metadata
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        # Get unique DOIs
        dois = set(metadata['dois'])
        print(f"Found {len(dois)} unique DOIs")
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            for doi in sorted(dois):
                f.write(f"{doi}\n")
                
        print(f"DOIs saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing metadata: {e}")

def main():
    metadata_file = "recursive_paragraph_embeddings_metadata.pkl"
    output_file = "recursive_reference_dois.txt"
    
    save_openai_dois(metadata_file, output_file)

if __name__ == "__main__":
    main()