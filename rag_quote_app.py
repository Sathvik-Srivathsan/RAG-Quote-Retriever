import streamlit as st
import pandas as pd
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import os
import json # For potential structured output, though Streamlit handles dicts well

# --- Configuration ---
DATASET_NAME = "Abirate/english_quotes"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Good general-purpose sentence transformer
LLM_MODEL_NAME = 'google/flan-t5-small' # Small, manageable, instruction-tuned

# --- File Paths for Preprocessed Data and Index ---
# Create a directory for artifacts if it doesn't exist
ARTIFACTS_DIR = "rag_quote_artifacts"
if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)

PROCESSED_DATA_PATH = os.path.join(ARTIFACTS_DIR, "processed_quotes.json") # Store as JSON list of dicts
FAISS_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "quotes_faiss.index")
EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "quote_embeddings.npy")


# --- 1. Data Preparation ---
def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    # Add more cleaning if needed, e.g., removing special characters not part of quotes
    return text

def process_tags(tags_list):
    """Cleans and normalizes a list of tags."""
    if not isinstance(tags_list, list):
        return []
    cleaned_tags = [clean_text(str(tag)) for tag in tags_list if isinstance(tag, str) and clean_text(str(tag))]
    return list(set(cleaned_tags)) # Remove duplicates

def load_and_preprocess_data():
    """Loads dataset, preprocesses it, and creates combined text for embeddings."""
    st.write(f"Loading '{DATASET_NAME}' dataset from Hugging Face...")
    try:
        dataset = load_dataset(DATASET_NAME)
        # Assuming the dataset has a 'train' split, common in Hugging Face datasets
        if 'train' in dataset:
            df = pd.DataFrame(dataset['train'])
        else: # Fallback if structure is different (e.g., dataset is the split itself)
            df = pd.DataFrame(dataset)
        st.write(f"Dataset loaded with {len(df)} entries.")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.error("Please check your internet connection and Hugging Face dataset availability.")
        return None

    # --- Data Cleaning and Structuring ---
    # Select relevant columns (adjust if dataset structure is different)
    # Common columns for quote datasets are 'quote', 'author', 'tags'
    if 'quote' not in df.columns:
        st.error("Critical 'quote' column not found in the dataset.")
        return None

    df.rename(columns={'quote': 'original_quote'}, inplace=True) # Keep original for display

    # Handle missing values
    df['author'] = df['author'].fillna("Unknown Author").astype(str)
    # Tags might be tricky; ensure it's a list of strings
    if 'tags' not in df.columns:
        df['tags'] = pd.Series([[] for _ in range(len(df))]) # Add empty list if no tags column
    else:
        df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [])

    df['cleaned_quote'] = df['original_quote'].apply(clean_text)
    df['cleaned_author'] = df['author'].apply(clean_text)
    df['cleaned_tags'] = df['tags'].apply(process_tags)

    # Drop rows if cleaned_quote is empty after cleaning
    df.dropna(subset=['cleaned_quote'], inplace=True)
    df = df[df['cleaned_quote'] != '']

    # Create a combined text for semantic search embedding
    # This helps capture context from quote, author, and tags
    def create_combined_text(row):
        tags_str = ", ".join(row['cleaned_tags'])
        return f"Quote: {row['cleaned_quote']} Author: {row['cleaned_author']}. Tags: {tags_str}."

    df['combined_text_for_embedding'] = df.apply(create_combined_text, axis=1)

    # Convert DataFrame to list of dictionaries for easier JSON serialization
    processed_data_list = df[['original_quote', 'author', 'cleaned_tags', 'combined_text_for_embedding']].to_dict(orient='records')

    with open(PROCESSED_DATA_PATH, 'w') as f:
        json.dump(processed_data_list, f)

    st.success(f"Data preprocessing complete. {len(processed_data_list)} quotes processed and saved.")
    return processed_data_list

# --- 2. Model Fine-Tuning (Using Pre-trained for Embeddings) ---
@st.cache_resource # Cache the model loading
def load_embedding_model():
    st.write(f"Loading sentence embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        st.success("Sentence embedding model loaded.")
        return model
    except Exception as e:
        st.error(f"Error loading sentence embedding model: {e}")
        return None

def generate_embeddings(data_list, model):
    """Generates embeddings for the combined text in the data."""
    if not data_list or model is None:
        return None

    st.write("Generating embeddings for quotes... This may take some time.")
    combined_texts = [item['combined_text_for_embedding'] for item in data_list]

    # Generate embeddings in batches if dataset is very large (optional for this size)
    embeddings = model.encode(combined_texts, show_progress_bar=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    st.success(f"Embeddings generated and saved. Shape: {embeddings.shape}")
    return embeddings

# --- 3. Build the RAG Pipeline (FAISS Index) ---
def build_faiss_index(embeddings):
    """Builds and saves a FAISS index."""
    if embeddings is None or embeddings.shape[0] == 0:
        st.error("No embeddings to build FAISS index.")
        return None

    st.write("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # Using L2 distance
    index.add(embeddings.astype('float32')) # FAISS expects float32
    faiss.write_index(index, FAISS_INDEX_PATH)
    st.success(f"FAISS index built and saved. Total vectors: {index.ntotal}")
    return index

@st.cache_resource
def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        st.write("Loading existing FAISS index...")
        return faiss.read_index(FAISS_INDEX_PATH)
    return None

@st.cache_data # Cache the processed data loading
def load_processed_data():
    if os.path.exists(PROCESSED_DATA_PATH):
        st.write("Loading existing processed quote data...")
        with open(PROCESSED_DATA_PATH, 'r') as f:
            return json.load(f)
    return None

# --- Retriever Function ---
def retrieve_quotes(query, model, index, data_list, top_k=5):
    """Retrieves top_k relevant quotes using FAISS."""
    if model is None or index is None or not data_list:
        return [], []
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)

    results = []
    scores = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        dist = distances[0][i]
        if 0 <= idx < len(data_list): # Ensure index is valid
            results.append(data_list[idx])
            scores.append(1 / (1 + dist)) # Example similarity score (higher is better)
    return results, scores

# --- LLM for Generation ---
@st.cache_resource
def load_llm_and_tokenizer():
    st.write(f"Loading LLM and Tokenizer: {LLM_MODEL_NAME}...")
    try:
        tokenizer = T5Tokenizer.from_pretrained(LLM_MODEL_NAME)
        llm_model = T5ForConditionalGeneration.from_pretrained(LLM_MODEL_NAME)
        st.success("LLM and Tokenizer loaded.")
        return tokenizer, llm_model
    except Exception as e:
        st.error(f"Error loading LLM/Tokenizer: {e}")
        return None, None

def generate_response_with_llm(query, retrieved_quotes_data, tokenizer, llm_model):
    """Generates a response using LLM based on query and retrieved quotes."""
    if not retrieved_quotes_data or tokenizer is None or llm_model is None:
        return "Could not generate a response due to missing components or no relevant quotes found."

    context_str = "\n\n".join([
        f"Quote: \"{item['original_quote']}\" - Author: {item['author']}. (Tags: {', '.join(item['cleaned_tags'])})"
        for item in retrieved_quotes_data
    ])

    # Improved prompt for flan-t5 style models
    prompt = f"""Based on the following quotes:

{context_str}

Answer the question: "{query}"

If the quotes directly answer the question, use their information.
If not, summarize what the quotes are about in relation to the question.
Provide a concise answer.
Answer:"""

    st.write("--- LLM Prompt (for debugging) ---")
    st.text_area("Prompt sent to LLM:", prompt, height=150)
    st.write("--- End LLM Prompt ---")

    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True) # Increased max_length for more context
        outputs = llm_model.generate(
            inputs.input_ids,
            max_length=150,  # Max length of the generated summary/answer
            min_length=20,   # Min length
            num_beams=4,     # Beam search for better quality
            early_stopping=True,
            no_repeat_ngram_size=2 # Avoid repetition
        )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        st.error(f"Error during LLM generation: {e}")
        return "Error generating response."


# --- Main Function for Initial Setup ---
def initial_setup():
    """Performs initial data loading, processing, embedding, and indexing if needed."""
    # Check if all necessary artifact files exist
    all_artifacts_exist = (
            os.path.exists(PROCESSED_DATA_PATH) and
            os.path.exists(EMBEDDINGS_PATH) and # Check for embeddings too
            os.path.exists(FAISS_INDEX_PATH)
    )

    if all_artifacts_exist:
        st.sidebar.success("All preprocessed artifacts found and loaded.")
        return True # Indicate setup is complete or artifacts loaded

    st.sidebar.warning("One or more artifacts not found. Starting initial data setup...")

    # 1. Load and preprocess data
    with st.spinner("Step 1/3: Loading and preprocessing data..."):
        processed_data = load_and_preprocess_data()
    if processed_data is None:
        st.sidebar.error("Data preprocessing failed. Cannot continue.")
        return False

    # 2. Load embedding model and generate embeddings
    with st.spinner("Step 2/3: Loading embedding model and generating embeddings..."):
        embedding_model = load_embedding_model()
        if embedding_model is None:
            st.sidebar.error("Embedding model loading failed. Cannot continue.")
            return False
        embeddings = generate_embeddings(processed_data, embedding_model)
    if embeddings is None:
        st.sidebar.error("Embedding generation failed. Cannot continue.")
        return False

    # 3. Build FAISS index
    with st.spinner("Step 3/3: Building FAISS index..."):
        build_faiss_index(embeddings)

    st.sidebar.success("Initial setup complete! Artifacts created.")
    st.balloons()
    return True


# --- 5. Streamlit Application ---
def main():
    st.set_page_config(page_title="Semantic Quote Retriever with RAG", layout="wide")
    st.title("📚 Semantic Quote Retriever with RAG")
    st.markdown("""
    Enter a natural language query to find relevant quotes. 
    The system uses sentence embeddings for retrieval and an LLM to generate a summary based on the findings.
    """)

    # Perform initial setup (downloads data, creates index if not exists)
    # This will run once if artifacts are missing.
    with st.sidebar:
        st.header("Setup Status")
        if 'setup_done' not in st.session_state:
            st.session_state.setup_done = False

        if not st.session_state.setup_done:
            if initial_setup():
                st.session_state.setup_done = True
            else:
                st.error("Critical error during initial setup. Please check logs and try again.")
                return # Stop app if setup fails
        else:
            st.success("Setup complete. Artifacts are ready.")

    if not st.session_state.get('setup_done', False):
        st.warning("Please wait for the initial setup to complete (check sidebar).")
        return

    # Load necessary components (models, index, data) using caching
    embedding_model = load_embedding_model()
    faiss_index = load_faiss_index()
    processed_data_list = load_processed_data()
    tokenizer, llm_model = load_llm_and_tokenizer()

    if not all([embedding_model, faiss_index, processed_data_list, tokenizer, llm_model]):
        st.error("One or more critical components could not be loaded. The app cannot function.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.header("Example Queries:")
    example_queries = [
        "Quotes about hope by Oscar Wilde",
        "What did Einstein say about insanity?",
        "Motivational quotes about accomplishment",
        "All Oscar Wilde quotes with humor",
        "Life lessons from famous authors",
        "Quotes about courage by women authors"
    ]
    for ex_query in example_queries:
        if st.sidebar.button(ex_query, key=f"ex_{ex_query}"):
            st.session_state.query_input = ex_query

    query = st.text_input("Enter your query about quotes:",
                          key="query_input",
                          value=st.session_state.get("query_input", "What are some inspiring quotes about perseverance?"))

    top_k_retrieval = st.slider("Number of quotes to retrieve (for LLM context):", min_value=1, max_value=10, value=3)

    if st.button("Search for Quotes", type="primary"):
        if not query:
            st.warning("Please enter a query.")
        else:
            st.markdown("---")
            st.subheader("🔍 Search Results")

            with st.spinner("Retrieving relevant quotes..."):
                retrieved_quotes, similarity_scores = retrieve_quotes(query, embedding_model, faiss_index, processed_data_list, top_k=top_k_retrieval)

            if not retrieved_quotes:
                st.warning("No relevant quotes found for your query.")
            else:
                st.subheader("💬 LLM Generated Summary/Answer")
                with st.spinner("Generating response with LLM... This might take a moment."):
                    llm_response = generate_response_with_llm(query, retrieved_quotes, tokenizer, llm_model)
                st.markdown(llm_response)
                st.markdown("---")

                st.subheader(f"Top {len(retrieved_quotes)} Retrieved Quotes (Context for LLM):")

                # Prepare structured JSON-like output for display
                output_data = {
                    "query": query,
                    "llm_summary": llm_response,
                    "retrieved_entries": []
                }

                for i, (item, score) in enumerate(zip(retrieved_quotes, similarity_scores)):
                    entry = {
                        "rank": i + 1,
                        "quote": item['original_quote'],
                        "author": item['author'],
                        "tags": item['cleaned_tags'],
                        "similarity_score_to_query": f"{score:.4f}" # (approximate)
                    }
                    output_data["retrieved_entries"].append(entry)

                    with st.expander(f"Quote #{i+1} by {item['author']} (Score: {score:.4f})", expanded=(i<3)): # Expand first 3
                        st.markdown(f"> {item['original_quote']}")
                        st.caption(f"Author: {item['author']}")
                        if item['cleaned_tags']:
                            st.caption(f"Tags: {', '.join(item['cleaned_tags'])}")

                st.markdown("---")
                st.subheader("Structured JSON-like Response (for programmatic use):")
                st.json(output_data)

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note on Fine-Tuning & Evaluation:**
    - The sentence embedding model used here is pre-trained. Fine-tuning it on a quote-specific similarity task could improve retrieval relevance but requires a labeled dataset.
    - RAG evaluation (e.g., with RAGAS) is a complex process involving metrics like faithfulness, answer relevancy, and context precision. It typically requires a benchmark dataset of queries and ideal responses/contexts. This app focuses on the RAG pipeline implementation.
    """)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Models Used:**\n- Embedding: `{EMBEDDING_MODEL_NAME}`\n- LLM: `{LLM_MODEL_NAME}`")


if __name__ == "__main__":
    main()