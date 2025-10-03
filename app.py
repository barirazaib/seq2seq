# app.py
import streamlit as st
import torch
import os
import requests
from model import load_model, transliterate_text

# Set page configuration
st.set_page_config(
    page_title="Text Transliteration App",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 1rem;
    }
    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def download_model_files():
    """Download model files from GitHub automatically on startup"""
    files = {
        'best_seq2seq_joint.pth': 'https://github.com/barirazaib/seq2seq/raw/main/best_seq2seq_joint.pth',
        'joint_char.model': 'https://github.com/barirazaib/seq2seq/raw/main/joint_char.model'
    }
    
    downloaded_all = True
    download_status = {}
    
    for filename, url in files.items():
        if not os.path.exists(filename):
            try:
                # Show download progress
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                downloaded_size = 0
                
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        f.write(chunk)
                        downloaded_size += len(chunk)
                
                download_status[filename] = {
                    'status': 'success',
                    'size': downloaded_size
                }
                
            except Exception as e:
                download_status[filename] = {
                    'status': 'error',
                    'error': str(e)
                }
                downloaded_all = False
        else:
            # File already exists
            file_size = os.path.getsize(filename)
            download_status[filename] = {
                'status': 'exists',
                'size': file_size
            }
    
    return downloaded_all, download_status

@st.cache_resource
def initialize_model():
    """Initialize the model with automatic file downloading"""
    # First, ensure model files are available
    downloaded_all, download_status = download_model_files()
    
    if not downloaded_all:
        return None, None, None, download_status
    
    try:
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        model, sp = load_model(
            model_path="best_seq2seq_joint.pth",
            sp_model_path="joint_char.model",
            device=device
        )
        
        return model, sp, device, download_status
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, download_status

def show_download_status(download_status):
    """Show download status in the sidebar"""
    st.sidebar.markdown("### 📁 Model Files Status")
    
    for filename, status_info in download_status.items():
        if status_info['status'] == 'exists':
            st.sidebar.success(f"✅ {filename} ({status_info['size'] // 1024} KB)")
        elif status_info['status'] == 'success':
            st.sidebar.success(f"✅ Downloaded {filename} ({status_info['size'] // 1024} KB)")
        elif status_info['status'] == 'error':
            st.sidebar.error(f"❌ {filename}: {status_info['error']}")

def main():
    # Show loading spinner while initializing
    with st.spinner("🚀 Initializing transliteration app... Please wait."):
        model, sp, device, download_status = initialize_model()
    
    # Header
    st.markdown('<h1 class="main-header">🔤 Text Transliteration System</h1>', unsafe_allow_html=True)
    
    # Sidebar with download status
    show_download_status(download_status)
    
    st.sidebar.markdown("---")
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a Sequence-to-Sequence (Seq2Seq) model with LSTM layers "
        "for text transliteration. The model is trained on character-level representations "
        "using SentencePiece tokenization."
    )
    
    # Check if model loaded successfully
    if model is None:
        st.error("""
        ❌ Failed to initialize the model. 
        
        **Possible solutions:**
        1. Check your internet connection and reload the app
        2. The model files might be temporarily unavailable
        3. Try refreshing the page in a few moments
        
        If the problem persists, please check the GitHub repository for updates.
        """)
        
        # Show retry button
        if st.button("🔄 Retry Initialization"):
            st.rerun()
        return
    
    # Show success message
    st.success(f"✅ Model loaded successfully on **{device}**!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Input Text")
        
        # Input options
        input_method = st.radio(
            "Choose input method:",
            ["Text Input", "Example Texts"],
            horizontal=True
        )
        
        input_text = ""
        if input_method == "Text Input":
            input_text = st.text_area(
                "Enter text to transliterate:",
                placeholder="Type your text here...",
                height=150,
                key="input_text"
            )
        else:
            example_options = {
                "Hello world": "Hello world",
                "How are you?": "How are you?",
                "Machine learning": "Machine learning", 
                "Natural language processing": "Natural language processing",
                "Test sentence": "Test sentence"
            }
            selected_example = st.selectbox(
                "Choose an example:",
                list(example_options.keys()),
                key="example_selector"
            )
            input_text = example_options[selected_example]
            st.text_area(
                "Selected example:", 
                value=input_text, 
                height=100, 
                disabled=True,
                key="example_display"
            )
    
    with col2:
        st.subheader("📤 Output")
        
        if input_text and st.button("🚀 Transliterate", use_container_width=True, key="transliterate_btn"):
            with st.spinner("Transliterating..."):
                try:
                    # Perform transliteration
                    output_text = transliterate_text(model, input_text, sp, device)
                    
                    # Display result
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown("**Transliterated Text:**")
                    st.success(output_text)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Copy to clipboard functionality
                    st.code(output_text, language="text")
                    
                    # Download button for result
                    st.download_button(
                        label="📥 Download Result",
                        data=output_text,
                        file_name="transliterated_text.txt",
                        mime="text/plain",
                        key="download_btn"
                    )
                    
                except Exception as e:
                    st.error(f"Error during transliteration: {str(e)}")
        
        elif not input_text:
            st.info("👆 Enter some text on the left and click the transliterate button!")
    
    # Additional information
    st.markdown("---")
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.markdown("### ℹ️ Model Info")
        st.markdown("""
        - **Architecture**: Seq2Seq with LSTM
        - **Encoder**: Bidirectional LSTM  
        - **Decoder**: LSTM
        - **Tokenization**: SentencePiece
        """)
    
    with col_info2:
        st.markdown("### ⚙️ Technical Details")
        st.markdown(f"""
        - **Embedding Dim**: 256
        - **Hidden Dim**: 256
        - **Layers**: 2
        - **Dropout**: 0.3
        - **Device**: {device}
        - **Vocab Size**: {sp.get_piece_size()}
        """)
    
    with col_info3:
        st.markdown("### 📝 Usage Tips")
        st.markdown("""
        - Enter text in the input box
        - Click the transliterate button
        - View results in the output section
        - Use examples to test the model
        - Results are character-level transliterations
        """)

if __name__ == "__main__":
    main()
