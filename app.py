import os
import ctypes
import pickle
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory, session
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer
import faiss

# Load environment variables from .env file
load_dotenv()

# Create Flask app with explicit template folder
app = Flask(__name__, 
            template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates')),
            static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), 'static')))

app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))

class NanoBERTSearchEngine:
    def __init__(self, 
                 hf_api_key=None, 
                 model_name="Flamenco43/NanoBERT-V2", 
                 checkpoint_file="recursive_paragraph_embeddings.faiss", 
                 metadata_file="recursive_paragraph_metadata.pkl"):
        # Hugging Face API Key (can be passed as environment variable)
        self.hf_api_key = hf_api_key or os.getenv('HF_API_KEY')
        if not self.hf_api_key:
            raise ValueError("Hugging Face API Key is required. Set HF_API_KEY environment variable.")
        
        print(f"\nInitializing with model: {model_name}")
        self.model_name = model_name
        self.inference_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        # Load FAISS index and metadata
        print("\nLoading FAISS index...")
        self.index = faiss.read_index(checkpoint_file)
        print(f"FAISS index type: {type(self.index)}")
        print(f"FAISS index description: {self.index}")
        print(f"Number of vectors in index: {self.index.ntotal}")
        print(f"Vector dimension: {self.index.d}")
        
        print("\nLoading metadata...")
        with open(metadata_file, "rb") as f:
            raw_metadata = pickle.load(f)
        
        print("Raw metadata keys:", raw_metadata.keys())
        print(f"Raw metadata types: {[(k, type(v), len(v) if hasattr(v, '__len__') else 'N/A') for k, v in raw_metadata.items()]}")
        
        # Convert metadata to the expected format
        self.metadata = {}
        titles = raw_metadata.get('titles', [])
        dois = raw_metadata.get('dois', [])
        paragraph_ids = raw_metadata.get('paragraph_ids', [])
        chunk_ids = raw_metadata.get('chunk_ids', [])
        
        print(f"\nTitles length: {len(titles)}")
        print(f"DOIs length: {len(dois)}")
        print(f"Paragraph IDs length: {len(paragraph_ids)}")
        print(f"Chunk IDs length: {len(chunk_ids)}")
        
        for idx in range(len(titles)):
            self.metadata[idx] = {
                'title': titles[idx] if idx < len(titles) else "Title not available",
                'doi': dois[idx] if idx < len(dois) else "DOI not available",
                'paragraph_id': paragraph_ids[idx] if idx < len(paragraph_ids) else "Paragraph ID not available",
                'chunk_id': chunk_ids[idx] if idx < len(chunk_ids) else "Chunk ID not available"
            }
        
        print(f"\nProcessed {len(self.metadata)} metadata entries")
        
        # Initialize tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded successfully")
        
        # Get dimension from the index
        self.embedding_dim = self.index.d
        print(f"FAISS index dimension: {self.embedding_dim}")

    def encode_text(self, text):
        """
        Vectorize text using Hugging Face Inference API
        """
        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                self.inference_url, 
                headers=headers, 
                json={"inputs": text, "options": {"wait_for_model": True}}
            )
            response.raise_for_status()
            
            # Extract vector from response
            vector = response.json()
            vector_array = np.array(vector, dtype=np.float32)
            
            # Handle different output formats
            if len(vector_array.shape) == 3:  # [batch_size, sequence_length, hidden_size]
                # Mean pooling over sequence length
                vector_array = np.mean(vector_array, axis=1)
            elif len(vector_array.shape) == 2:  # [sequence_length, hidden_size]
                # Mean pooling if needed
                vector_array = np.mean(vector_array, axis=0, keepdims=True)
            
            print(f"Original vector shape: {vector_array.shape}")
            
            # If dimensions don't match, try mean pooling or truncation
            if vector_array.shape[-1] != self.embedding_dim:
                if vector_array.shape[-1] > self.embedding_dim:
                    # Truncate to match FAISS index dimensions
                    vector_array = vector_array[..., :self.embedding_dim]
                else:
                    # Pad with zeros if vector is too small (shouldn't happen normally)
                    pad_width = ((0, 0), (0, self.embedding_dim - vector_array.shape[-1]))
                    vector_array = np.pad(vector_array, pad_width, mode='constant')
            
            print(f"Final vector shape: {vector_array.shape}")
            return vector_array.reshape(1, -1)
            
        except Exception as e:
            print(f"Error in encode_text: {str(e)}")
            print(f"Response content: {response.content if 'response' in locals() else 'No response'}")
            raise

    def chunk_text(self, text, max_length=512):
        """
        Chunk text using the tokenizer's max length constraint
        """
        try:
            # First tokenize without tensors
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                return_tensors=None
            )
            
            # Calculate number of chunks needed
            chunk_size = max_length - 2  # Account for special tokens
            chunks = []
            
            # Create chunks with overlap
            stride = int(chunk_size * 0.8)  # 20% overlap
            for i in range(0, len(tokens), stride):
                chunk_tokens = tokens[i:i + chunk_size]
                
                # Decode chunk back to text
                chunk_text = self.tokenizer.decode(
                    chunk_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                if chunk_text.strip():  # Only add non-empty chunks
                    chunks.append(chunk_text)
            
            return chunks if chunks else [text[:1000]]  # Fallback to simple truncation
            
        except Exception as e:
            print(f"Error in chunk_text: {str(e)}")
            return [text[:1000]]  # Return truncated text as fallback

    def get_chunk_text(self, paragraph_id, chunk_id):
        """Get the specific chunk from a paragraph"""
        try:
            paragraph = self.paragraphs[paragraph_id]
            chunks = self.chunk_text(paragraph)
            
            if not chunks:
                return "No valid chunks found"
                
            if 0 <= chunk_id < len(chunks):
                return chunks[chunk_id]
            
            # If chunk_id is out of range, return the first chunk
            return chunks[0] if chunks else "Chunk not found"
            
        except Exception as e:
            print(f"Error in get_chunk_text: {str(e)}")
            return f"Error retrieving chunk: {str(e)}"

    def search(self, query, top_k=5):
        try:
            query_vector = self.encode_text(query)
            
            # Get more results initially
            D, I = self.index.search(query_vector, top_k * 5)  # Increased multiplier
            
            print("\nRaw search results:")
            for d, i in zip(D[0], I[0]):
                print(f"Distance: {d}, Index: {i}")
            
            # Create a dictionary to track unique results by DOI and paragraph
            unique_results = {}
            
            # Process results and remove duplicates
            for distance, idx in zip(D[0], I[0]):
                if idx == -1:
                    print(f"Skipping invalid index: {idx}")
                    continue
                    
                idx = int(idx)
                
                if idx not in self.metadata:
                    print(f"Warning: Index {idx} not found in metadata")
                    continue
                
                metadata_entry = self.metadata[idx]
                doi = metadata_entry['doi']
                paragraph_id = metadata_entry['paragraph_id']
                
                # Create a unique key combining DOI and paragraph_id
                unique_key = f"{doi}_{paragraph_id}"
                
                print(f"\nProcessing result:")
                print(f"Index: {idx}")
                print(f"Distance: {distance}")
                print(f"DOI: {doi}")
                print(f"Paragraph ID: {paragraph_id}")
                print(f"Title: {metadata_entry['title']}")
                
                # Only keep if it's a new DOI or a better match for this paragraph
                if unique_key not in unique_results or distance < unique_results[unique_key]['distance']:
                    unique_results[unique_key] = {
                        'distance': float(distance),
                        'title': metadata_entry['title'],
                        'doi': doi,
                        'paragraph_id': paragraph_id,
                        'chunk_id': metadata_entry['chunk_id']
                    }
                
                # Break if we have enough unique results
                if len(unique_results) >= top_k:
                    # Check if we have enough different DOIs
                    unique_dois = len(set(r['doi'] for r in unique_results.values()))
                    if unique_dois >= top_k:
                        break
            
            # Convert dictionary to list and sort by distance
            results = list(unique_results.values())
            results.sort(key=lambda x: x['distance'])
            
            # Group by DOI and take the best result for each
            doi_results = {}
            for result in results:
                doi = result['doi']
                if doi not in doi_results or result['distance'] < doi_results[doi]['distance']:
                    doi_results[doi] = result
            
            # Final results sorted by distance
            final_results = list(doi_results.values())
            final_results.sort(key=lambda x: x['distance'])
            final_results = final_results[:top_k]
            
            print(f"\nFinal results:")
            for r in final_results:
                print(f"Distance: {r['distance']}, DOI: {r['doi']}, Paragraph: {r['paragraph_id']}")
            
            return final_results
            
        except Exception as e:
            print(f"Error in search method: {str(e)}")
            raise

# Initialize search engine (do this once to avoid reloading on every request)
search_engine = NanoBERTSearchEngine()

@app.route('/')
def index():
    try:
        # Get FAISS index stats from the search_engine instance
        faiss_count = search_engine.index.ntotal
        vector_dim = search_engine.index.d
        using_recursive = session.get('use_recursive', True)
        # Get unique DOIs count
        unique_dois = len(set(doi for metadata in search_engine.metadata.values() for doi in [metadata['doi']]))
    except Exception as e:
        print(f"Error getting FAISS stats: {e}")
        faiss_count = 0
        vector_dim = 0
        using_recursive = True
        unique_dois = 0
    
    return render_template('index.html', 
                         faiss_count=faiss_count,
                         vector_dim=vector_dim,
                         using_recursive=using_recursive,
                         unique_dois=unique_dois)

@app.route('/search', methods=['POST'])
def search_documents():
    print("\n=== New Search Request ===")
    print(f"Request Method: {request.method}")
    print(f"Request Headers: {dict(request.headers)}")
    
    try:
        # Get and log raw request data
        raw_data = request.get_data()
        print(f"Raw request data: {raw_data}")
        
        # Get JSON data
        data = request.get_json()
        print(f"Parsed JSON data: {data}")
        
        if not data:
            print("Error: No JSON data provided")
            return jsonify({"error": "No JSON data provided"}), 400
        
        query = data.get('query', '')
        top_k = data.get('top_k', 5)
        print(f"Query: '{query}', top_k: {top_k}")

        if not query:
            print("Error: No query provided")
            return jsonify({"error": "No query provided"}), 400

        # Perform search
        try:
            print("Starting search...")
            results = search_engine.search(query, top_k)
            print(f"Search completed. Found {len(results)} results")
            print(f"Results: {results}")
            
            return jsonify({
                "query": query,
                "results": results
            })

        except Exception as search_error:
            print(f"Search error: {str(search_error)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                "error": "Search failed",
                "details": str(search_error)
            }), 500

    except Exception as e:
        print(f"Request handling error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/debug-static/<path:filename>')
def debug_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.after_request
def add_header(response):
    if response.headers['Content-Type'] == 'application/javascript':
        response.headers['Content-Type'] = 'text/babel'
    return response

@app.route('/toggle_embeddings', methods=['POST'])
def toggle_embeddings():
    """Toggle between recursive and standard embeddings"""
    try:
        # Add debug logging
        print("\n=== Toggle Embeddings Request ===")
        print(f"Current session state: {dict(session)}")
        
        # Toggle the current setting
        current = session.get('use_recursive', True)
        session['use_recursive'] = not current
        
        print(f"New session state: {dict(session)}")
        print(f"Switching to {'recursive' if session['use_recursive'] else 'standard'} embeddings")
        
        # Reinitialize search engine with new files
        global search_engine
        if session['use_recursive']:
            print("Loading recursive embeddings...")
            search_engine = NanoBERTSearchEngine(
                checkpoint_file="recursive_paragraph_embeddings.faiss",
                metadata_file="recursive_paragraph_metadata.pkl"
            )
        else:
            print("Loading standard embeddings...")
            search_engine = NanoBERTSearchEngine(
                checkpoint_file="paragraph_embeddings.faiss",
                metadata_file="paragraph_metadata.pkl"
            )
        
        # Get unique DOIs count for new engine
        unique_dois = len(set(doi for metadata in search_engine.metadata.values() for doi in [metadata['doi']]))
        
        response_data = {
            "success": True,
            "using_recursive": session['use_recursive'],
            "vector_count": search_engine.index.ntotal,
            "vector_dim": search_engine.index.d,
            "unique_dois": unique_dois
        }
        print(f"Response data: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Toggle error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# For Coolify deployment
if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 3000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    app.run(
        host=host,
        port=port,
        debug='true'
    )