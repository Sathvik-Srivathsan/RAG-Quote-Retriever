# RAG-Quote-Retriever
# Semantic Quote Retriever with RAG

This project is a Streamlit web application that allows users to search for quotes based on their semantic meaning (not just keywords). It uses a Retrieval Augmented Generation (RAG) pipeline:

1. **Retrieval:** Finds relevant quotes using sentence embeddings and a FAISS vector index.
2. **Generation:** Uses a Large Language Model (LLM) to generate a summary or answer based on the retrieved quotes and the user's query.

## Features

- Loads quotes from a Hugging Face dataset (`Abirate/english_quotes`).
- Cleans and preprocesses text data.
- Generates sentence embeddings using `all-MiniLM-L6-v2`.
- Builds a FAISS index for efficient similarity search.
- Retrieves top-k relevant quotes for a user query.
- Uses `google/flan-t5-small` to generate a contextual response.
- Interactive web interface built with Streamlit.
- Caches processed data, embeddings, index, and models for faster subsequent runs.

## How it Works (Simplified for Beginners)

Imagine you have a huge library of quotes.

1. **Understanding Quotes:** We first teach a computer to understand the *meaning* of each quote. It does this by converting each quote into a special list of numbers (called an "embedding"). Quotes with similar meanings will have similar lists of numbers.
2. **Indexing for Speed:** To quickly find similar quotes later, we put all these number-lists into a special, super-fast address book (this is the "FAISS index").
3. **Your Query:** When you ask a question (your "query"), we also convert your question into a similar list of numbers.
4. **Finding Matches (Retrieval):** We use the FAISS address book to find the quotes whose number-lists are most similar to your question's number-list. These are the "retrieved quotes."
5. **Smart Answers (Generation):** We then take your original question and the most relevant quotes we found, and give them to a very smart "Language Model" (like a super-powered chatbot). This model reads everything and writes a nice answer or summary for you. This is "Retrieval Augmented Generation" – we *retrieve* information to *augment* (help) the *generation* of an answer.

## Prerequisites

- **Python:** You need Python installed on your computer. Version 3.8 or newer is recommended.
    - **How to check if you have Python:** Open a terminal or command prompt and type `python --version` or `python3 --version`.
    - **How to install Python:** Go to [python.org](https://www.python.org/downloads/) and download the installer for your operating system (Windows, macOS, Linux). Make sure to check the box that says "Add Python to PATH" during installation if you're on Windows.
- **pip:** Python's package installer. It usually comes with Python.
    - **How to check:** In the terminal, type `pip --version` or `pip3 --version`.

## Setup and Installation

**1. Download the Code:**

- Save the Python script (the one you provided, let's call it `app.py`) to a folder on your computer. For example, create a folder named `QuoteRetriever`.

**2. Open a Terminal or Command Prompt:**

- **Windows:** Search for "Command Prompt" or "PowerShell".
- **macOS:** Search for "Terminal".
- **Linux:** Usually Ctrl+Alt+T or search for "Terminal".
- **Navigate to the project folder:** Use the `cd` command. If you saved `app.py` in `C:\Users\YourName\QuoteRetriever`, you'd type:
    
    ```
    cd C:\Users\YourName\QuoteRetriever
    
    ```
    
    (Adjust the path accordingly).
    

**3. Create a Virtual Environment (Recommended):**
This keeps the project's libraries separate from other Python projects.

- In your terminal (inside the `QuoteRetriever` folder), run:
    
    ```
    python -m venv venv
    
    ```
    
    (Or `python3 -m venv venv` if `python` doesn't work)
    
- Activate the virtual environment:
    - **Windows (Command Prompt):** `venv\Scripts\activate`
    - **Windows (PowerShell):** `venv\Scripts\Activate.ps1` (You might need to run `Set-ExecutionPolicy Unrestricted -Scope Process` first if you get an error).
    - **macOS/Linux:** `source venv/bin/activate`
    Your terminal prompt should now change to show `(venv)` at the beginning.

**4. Install Required Libraries:**
With your virtual environment activated, install the libraries using pip:

```
pip install streamlit pandas numpy faiss-cpu datasets sentence-transformers transformers torch

```

- `streamlit`: For the web app interface.
- `pandas`: For handling data in tables.
- `numpy`: For numerical operations (used by embeddings).
- `faiss-cpu`: For the similarity search index (CPU version).
- `datasets`: To download the quote dataset from Hugging Face.
- `sentence-transformers`: To convert sentences to embeddings.
- `transformers`: For the LLM (Flan-T5).
- `torch`: A core machine learning library, often a dependency for `transformers` and `sentence-transformers`.

**Why these libraries?**

- Each library provides specific tools we need. `streamlit` builds the interactive parts you see. `pandas` and `numpy` help organize and work with data. `faiss-cpu` is our super-fast address book. `datasets`, `sentence-transformers`, and `transformers` give us access to the pre-trained AI models for understanding and generating text. `torch` is the engine many of these AI models run on.

## Running the Application

**1. Ensure your virtual environment is activated** (see step 3 in Setup).

**2. Run the Streamlit app:**
In your terminal (still in the `QuoteRetriever` folder where `app.py` is located), type:

```
streamlit run app.py

```

(If you named your file something else, replace `app.py` with that filename).

**3. Open in Browser:**
Streamlit will usually automatically open a new tab in your web browser pointing to the application (e.g., `http://localhost:8501`). If not, the terminal will display the URL you can copy and paste into your browser.

**First Run - Patience is Key!**

- The **very first time** you run the app, it will take a while (several minutes, depending on your internet speed and computer).
- **What's happening:**
    1. It downloads the quote dataset.
    2. It downloads the sentence embedding model (`all-MiniLM-L6-v2`).
    3. It downloads the LLM and its tokenizer (`google/flan-t5-small`).
    4. It processes all quotes, generates embeddings for them, and builds the FAISS index.
- These downloaded models and processed files (called "artifacts") are saved in a folder named `rag_quote_artifacts` (created in the same directory as your script).
- **Subsequent runs will be much faster** because the app will load these saved artifacts instead of re-doing everything. You'll see messages in the Streamlit sidebar indicating this.

## Using the Application

1. **Query Input:** Type your question or topic about quotes into the text box labeled "Enter your query about quotes:".
2. **Number of Quotes:** Use the slider to select how many relevant quotes should be retrieved and used as context for the LLM.
3. **Search:** Click the "Search for Quotes" button.
4. **Results:**
    - **LLM Generated Summary/Answer:** The app will display a response generated by the LLM based on your query and the retrieved quotes.
    - **Retrieved Quotes:** Below the summary, you'll see the actual quotes that were found to be most relevant, along with their authors and similarity scores. These are expandable.
    - **JSON Output:** A structured JSON output of the query, summary, and retrieved entries is also provided for programmatic use or inspection.
5. **Example Queries:** The sidebar has some example queries you can click to auto-fill the input box.

## For Pycharm Community Users

1. **Open Project:**
    - Open Pycharm.
    - Click "Open" or "File" > "Open..."
    - Navigate to and select the folder where you saved `app.py` (e.g., `QuoteRetriever`).
2. **Set up Python Interpreter (Virtual Environment):**
    - Pycharm usually detects `venv` folders. If not, or to be sure:
    - Go to "File" > "Settings" (or "Pycharm" > "Preferences..." on macOS).
    - Navigate to "Project: [YourProjectName]" > "Python Interpreter".
    - Click the gear icon ⚙️ next to the Python Interpreter dropdown and select "Add...".
    - Choose "Existing environment".
    - For "Interpreter:", click the `...` button and navigate inside your project folder to `venv/Scripts/python.exe` (Windows) or `venv/bin/python` (macOS/Linux).
    - Click "OK" to save. Pycharm will now use this virtual environment.
3. **Install Packages (if not done via terminal):**
    - In Pycharm, open the "Terminal" tab at the bottom (or View > Tool Windows > Terminal).
    - Ensure your virtual environment is active in this terminal (you should see `(venv)` in the prompt). If not, activate it as described in Setup Step 3.
    - Run `pip install streamlit pandas numpy faiss-cpu datasets sentence-transformers transformers torch`.
4. **Run the App:**
    - In the Pycharm Terminal (with `venv` active and in the project root directory):
        
        ```
        streamlit run app.py
        
        ```
