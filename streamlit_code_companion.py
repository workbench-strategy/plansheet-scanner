"""
AI Code Companion - Streamlit Web Interface
Provides an interactive web interface for intelligent code analysis and retrieval.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.code_companion import AICodeCompanion, CodeCitation

# Page configuration
st.set_page_config(
    page_title="AI Code Companion",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'companion' not in st.session_state:
    st.session_state.companion = AICodeCompanion()
if 'indexed_files' not in st.session_state:
    st.session_state.indexed_files = set()

def main():
    """Main Streamlit application."""
    
    # Sidebar
    st.sidebar.title("🤖 AI Code Companion")
    st.sidebar.markdown("Intelligent code analysis and retrieval powered by ML/NN")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose a feature:",
        ["📊 Code Analysis", "🔍 Code Search", "💡 Code Suggestions", "📚 Code Citations", "⚙️ Settings"]
    )
    
    if page == "📊 Code Analysis":
        show_code_analysis()
    elif page == "🔍 Code Search":
        show_code_search()
    elif page == "💡 Code Suggestions":
        show_code_suggestions()
    elif page == "📚 Code Citations":
        show_code_citations()
    elif page == "⚙️ Settings":
        show_settings()

def show_code_analysis():
    """Show code analysis interface."""
    st.title("📊 Code Analysis")
    st.markdown("Analyze code files with intelligent highlighting and metrics")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a code file",
        type=['py', 'js', 'java', 'cpp', 'c', 'cs', 'php', 'rb', 'go', 'rs', 'swift', 'kt'],
        help="Upload a code file to analyze"
    )
    
    # Or paste code
    st.markdown("---")
    st.subheader("Or paste code directly:")
    pasted_code = st.text_area(
        "Paste your code here:",
        height=200,
        placeholder="def example_function():\n    return 'Hello, World!'"
    )
    
    if uploaded_file or pasted_code:
        if uploaded_file:
            # Save uploaded file temporarily
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, 'w') as f:
                f.write(uploaded_file.getvalue().decode())
            
            # Analyze file
            result = st.session_state.companion.analyze_file(temp_path)
            
            # Clean up
            os.remove(temp_path)
        else:
            # Analyze pasted code
            result = analyze_pasted_code(pasted_code)
        
        if 'error' not in result:
            display_analysis_results(result)
        else:
            st.error(f"Analysis failed: {result['error']}")

def analyze_pasted_code(code: str) -> Dict[str, Any]:
    """Analyze pasted code."""
    # Create a temporary analysis
    analyzer = st.session_state.companion.analyzer
    highlighter = st.session_state.companion.highlighter
    
    # Detect language (simple heuristic)
    language = 'python'  # Default
    if 'function' in code or 'var' in code or 'const' in code:
        language = 'javascript'
    elif 'public class' in code or 'private' in code:
        language = 'java'
    elif '#include' in code or 'int main' in code:
        language = 'cpp'
    
    complexity_metrics = analyzer.analyze_complexity(code, language)
    highlighted_code = highlighter.highlight_code(code, language)
    
    return {
        'file_path': 'pasted_code',
        'language': language,
        'metrics': complexity_metrics,
        'highlights': highlighted_code.highlights,
        'suggestions': highlighted_code.suggestions,
        'warnings': highlighted_code.warnings,
        'complexity_score': highlighted_code.complexity_score,
        'code': code
    }

def display_analysis_results(result: Dict[str, Any]):
    """Display analysis results in a nice format."""
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Language", result['language'].title())
    
    with col2:
        st.metric("Complexity Score", result['complexity_score'])
    
    with col3:
        if 'metrics' in result and 'lines_of_code' in result['metrics']:
            st.metric("Lines of Code", result['metrics']['lines_of_code'])
    
    with col4:
        if 'metrics' in result and 'function_count' in result['metrics']:
            st.metric("Functions", result['metrics']['function_count'])
    
    # Detailed metrics
    if 'metrics' in result:
        st.subheader("📈 Detailed Metrics")
        metrics_df = pd.DataFrame(list(result['metrics'].items()), columns=['Metric', 'Value'])
        st.dataframe(metrics_df, use_container_width=True)
    
    # Warnings and suggestions
    col1, col2 = st.columns(2)
    
    with col1:
        if result['warnings']:
            st.subheader("⚠️ Warnings")
            for warning in result['warnings']:
                st.warning(warning)
    
    with col2:
        if result['suggestions']:
            st.subheader("💡 Suggestions")
            for suggestion in result['suggestions']:
                st.info(suggestion)
    
    # Highlighted code
    st.subheader("🔍 Highlighted Code")
    
    if 'code' in result:
        code = result['code']
    else:
        # Read from file
        with open(result['file_path'], 'r') as f:
            code = f.read()
    
    # Create highlighted code display
    highlighted_html = create_highlighted_html(code, result['highlights'])
    st.markdown(highlighted_html, unsafe_allow_html=True)

def create_highlighted_html(code: str, highlights: List[Dict[str, Any]]) -> str:
    """Create HTML with syntax highlighting."""
    if not highlights:
        return f"<pre><code>{code}</code></pre>"
    
    # Sort highlights by start position
    highlights.sort(key=lambda x: x['start'])
    
    # Color mapping
    color_map = {
        'keywords': '#007acc',
        'strings': '#a31515',
        'comments': '#008000',
        'numbers': '#098658',
        'functions': '#d73a49',
        'classes': '#6f42c1',
        'potential_issue': '#ff8c00',
        'complexity_warning': '#ffd700'
    }
    
    html_parts = ['<pre><code style="background-color: #f6f8fa; padding: 16px; border-radius: 6px; font-family: monospace;">']
    
    last_end = 0
    for highlight in highlights:
        # Add text before highlight
        if highlight['start'] > last_end:
            html_parts.append(code[last_end:highlight['start']])
        
        # Add highlighted text
        color = color_map.get(highlight.get('color', 'black'), '#000000')
        highlighted_text = code[highlight['start']:highlight['end']]
        html_parts.append(f'<span style="color: {color}; font-weight: bold;">{highlighted_text}</span>')
        
        last_end = highlight['end']
    
    # Add remaining text
    if last_end < len(code):
        html_parts.append(code[last_end:])
    
    html_parts.append('</code></pre>')
    return ''.join(html_parts)

def show_code_search():
    """Show code search interface."""
    st.title("🔍 Code Search")
    st.markdown("Search for similar code using neural embeddings")
    
    # Index files
    st.subheader("📚 Index Files")
    col1, col2 = st.columns(2)
    
    with col1:
        directory = st.text_input(
            "Directory to index:",
            placeholder="/path/to/your/codebase",
            help="Enter a directory path to index all code files"
        )
        
        if st.button("Index Directory") and directory:
            if os.path.exists(directory):
                index_directory(directory)
            else:
                st.error("Directory not found!")
    
    with col2:
        uploaded_files = st.file_uploader(
            "Or upload files to index:",
            type=['py', 'js', 'java', 'cpp', 'c', 'cs'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                index_uploaded_file(uploaded_file)
    
    # Search interface
    st.markdown("---")
    st.subheader("🔍 Search Code")
    
    search_query = st.text_input(
        "Enter search query:",
        placeholder="def calculate_complexity",
        help="Enter code or function signature to search for"
    )
    
    top_k = st.slider("Number of results:", 1, 20, 5)
    
    if st.button("Search") and search_query:
        if not st.session_state.indexed_files:
            st.warning("No files indexed yet. Please index some files first.")
        else:
            search_results = st.session_state.companion.search_codebase(search_query, top_k)
            display_search_results(search_results)

def index_directory(directory: str):
    """Index all code files in a directory."""
    with st.spinner(f"Indexing files in {directory}..."):
        indexed_count = 0
        for file_path in Path(directory).rglob('*'):
            if file_path.is_file() and file_path.suffix in ['.py', '.js', '.java', '.cpp', '.c', '.cs']:
                try:
                    st.session_state.companion.analyze_file(str(file_path))
                    st.session_state.indexed_files.add(str(file_path))
                    indexed_count += 1
                except Exception as e:
                    st.error(f"Error indexing {file_path}: {e}")
        
        st.success(f"Indexed {indexed_count} files!")

def index_uploaded_file(uploaded_file):
    """Index an uploaded file."""
    try:
        # Save temporarily
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, 'w') as f:
            f.write(uploaded_file.getvalue().decode())
        
        st.session_state.companion.analyze_file(temp_path)
        st.session_state.indexed_files.add(uploaded_file.name)
        
        # Clean up
        os.remove(temp_path)
        
        st.success(f"Indexed {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error indexing {uploaded_file.name}: {e}")

def display_search_results(results: List[CodeCitation]):
    """Display search results."""
    if not results:
        st.info("No similar code found.")
        return
    
    st.subheader(f"📚 Found {len(results)} similar code snippets:")
    
    for i, citation in enumerate(results, 1):
        with st.expander(f"{i}. {citation.source_file} (similarity: {citation.similarity_score:.2f})"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Details:**")
                st.write(f"- Type: {citation.citation_type}")
                st.write(f"- Lines: {citation.snippet.start_line}-{citation.snippet.end_line}")
                st.write(f"- Language: {citation.snippet.language}")
                st.write(f"- Similarity: {citation.similarity_score:.2f}")
            
            with col2:
                st.write("**Code Context:**")
                st.code(citation.snippet.context, language=citation.snippet.language)

def show_code_suggestions():
    """Show code suggestions interface."""
    st.title("💡 Code Suggestions")
    st.markdown("Generate intelligent code suggestions using neural models")
    
    # Input method
    input_method = st.radio(
        "Choose input method:",
        ["Paste Code", "Upload File"]
    )
    
    if input_method == "Paste Code":
        context = st.text_area(
            "Enter code context:",
            height=200,
            placeholder="def process_data(data):\n    # Process the data\n    return result",
            help="Enter the code context for which you want suggestions"
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a code file:",
            type=['py', 'js', 'java', 'cpp', 'c', 'cs']
        )
        context = ""
        if uploaded_file:
            context = uploaded_file.getvalue().decode()
    
    language = st.selectbox(
        "Programming Language:",
        ['python', 'javascript', 'java', 'cpp', 'c', 'csharp']
    )
    
    if st.button("Generate Suggestions") and context:
        with st.spinner("Generating suggestions..."):
            suggestions = st.session_state.companion.generate_code_suggestions(context, language)
        
        st.subheader("🤖 Generated Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            with st.expander(f"Suggestion {i}"):
                st.code(suggestion, language=language)

def show_code_citations():
    """Show code citations interface."""
    st.title("📚 Code Citations")
    st.markdown("Find citations and references for code in your codebase")
    
    # Input method
    input_method = st.radio(
        "Choose input method:",
        ["Paste Code", "Upload File"]
    )
    
    if input_method == "Paste Code":
        code = st.text_area(
            "Enter code to find citations for:",
            height=200,
            placeholder="def calculate_complexity(data):\n    return len(data)",
            help="Enter the code for which you want to find citations"
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a code file:",
            type=['py', 'js', 'java', 'cpp', 'c', 'cs']
        )
        code = ""
        if uploaded_file:
            code = uploaded_file.getvalue().decode()
    
    if st.button("Find Citations") and code:
        if not st.session_state.indexed_files:
            st.warning("No files indexed yet. Please index some files first in the Code Search section.")
        else:
            with st.spinner("Finding citations..."):
                citations = st.session_state.companion.get_code_citations(code)
            
            display_search_results(citations)

def show_settings():
    """Show settings interface."""
    st.title("⚙️ Settings")
    st.markdown("Configure AI Code Companion settings")
    
    st.subheader("Model Configuration")
    
    # Model selection
    model_name = st.selectbox(
        "Embedding Model:",
        [
            "microsoft/codebert-base",
            "microsoft/codebert-base-mlm",
            "microsoft/graphcodebert-base"
        ],
        help="Choose the neural model for code embeddings"
    )
    
    if st.button("Update Model"):
        st.session_state.companion = AICodeCompanion(model_name)
        st.success("Model updated successfully!")
    
    st.subheader("Indexed Files")
    if st.session_state.indexed_files:
        st.write(f"Currently indexed: {len(st.session_state.indexed_files)} files")
        if st.button("Clear Index"):
            st.session_state.indexed_files.clear()
            st.session_state.companion.retriever.document_index.clear()
            st.success("Index cleared!")
    else:
        st.info("No files indexed yet.")
    
    st.subheader("About")
    st.markdown("""
    **AI Code Companion** is powered by:
    - **CodeBERT**: Neural code understanding model
    - **CodeT5**: Code generation model
    - **AST Analysis**: Static code analysis
    - **Neural Embeddings**: Semantic code similarity
    
    Built with ❤️ using PyTorch, Transformers, and Streamlit.
    """)

if __name__ == "__main__":
    main()