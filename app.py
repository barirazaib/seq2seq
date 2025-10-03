# app.py
import streamlit as st
import torch
import os
import requests
import time
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
    .stButton button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def download_with_retry(url, filename, max_retries=3):
    """Download file with retry mechanism and proper error handling"""
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Check if we got actual content (not HTML error page)
            content = response.content
            if len(content) < 1000 and (b'<!DOCTYPE html>' in content or b'<html>' in content):
                st.warning(f"Attempt {attempt + 1}: Got HTML instead of binary file. Retrying...")
                time.sleep(2)
                continue
            
            # Save file
            with open(filename, 'wb') as f:
                f.write(content)
            
            # Verify file size
            file_size = os.path.getsize(filename)
            if file_size == 0:
                st.warning(f"Attempt {attempt + 1}: Downloaded empty file. Retrying...")
                os.remove(filename)
                time.sleep(2)
                continue
                
            return True, file_size, None
            
        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
            st.warning(error_msg)
            if attempt < max_retries - 1:
                time.sleep(2)
    
    return False, 0, f"Failed after {max_retries} attempts"

@st.cache_resource
def download_model_files():
    """Download model files from GitHub automatically on startup"""
    files = {
        'best_seq2seq_joint.pth': 'https://github.com/barirazaib/seq2seq/raw/main/best_seq2seq_joint.pth',
        'joint_char.model': 'https://github.com/barirazaib/seq2seq/raw/main/joint_char.model'
    }
    
    # Alternative URLs in case main ones fail
    alternative_urls = {
        'best_seq2seq_joint.pth': [
            'https://github.com/barirazaib/seq2seq/raw/main/best_seq2seq_joint.pth',
            'https://raw.githubusercontent.com/barirazaib/seq2seq/main/best_seq2seq_joint.pth'
        ],
        'joint_char.model': [
            'https://github.com/barirazaib/seq2seq/raw/main/joint_char.model',
            'https://raw.githubusercontent.com/barirazaib/seq2seq/main/joint_char.model'
        ]
    }
    
    downloaded_all = True
    download_status = {}
    
    for filename, main_url in files.items():
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            if file_size > 1000:  # Reasonable minimum size
                download_status[filename] = {'status': 'exists', 'size': file_size}
                continue
            else:
                # Remove corrupted/small file
                os.remove(filename)
        
        # Try to download
        success = False
        file_size = 0
        error_msg = None
        
        # Try multiple URL variations
        for url in alternative_urls[filename]:
            st.info(f"Downloading {filename} from {url}...")
            success, file_size, error_msg = download_with_retry(url, filename)
            if success:
                break
        
        if success:
            download_status[filename] = {'status': 'success', 'size': file_size}
            st.toast(f"✅ Downloaded {filename} ({file_size // 1024} KB)", icon="✅")
        else:
            download_status[filename] = {'status': 'error', 'error': error_msg}
            downloaded_all = False
            st.toast(f"❌ Failed to download {filename}", icon="❌")
    
    return downloaded_all, download_status

@st.cache_resource
def initialize_model():
    """Initialize the model with automatic file downloading"""
    # First, ensure model files are available
    downloaded_all, download_status = download_model_files()
    
    if not downloaded_all:
        error_msg = "Failed to download model files. Please check your internet connection."
        return None, None, None, download_status, error_msg
    
    try:
        # Verify file sizes are reasonable
        model_size = os.path.getsize('best_seq2seq_joint.pth')
        tokenizer_size = os.path.getsize('joint_char.model')
        
        if model_size < 1000:  # Model file should be much larger
            error_msg = f"Model file seems too small ({model_size} bytes). It may be corrupted."
            return None, None, None, download_status, error_msg
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        model, sp = load_model(
            model_path="best_seq2seq_joint.pth",
            sp_model_path="joint_char.model",
            device=device
        )
        
        return model, sp, device, download_status, None
        
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        return None, None, None, download_status, error_msg

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
    # Header
    st.markdown('<h1 class="main-header">🔤 Text Transliteration System</h1>', unsafe_allow_html=True)
    
    # Initialize the app
    if 'initialized' not in st.session_state:
        with st.spinner("🚀 Initializing transliteration app... Downloading model files if needed..."):
            model, sp, device, download_status, error_msg = initialize_model()
        st.session_state.update({
            'model': model,
            'sp': sp,
            'device': device,
            'download_status': download_status,
            'error_msg': error_msg,
            'initialized': True
        })
    else:
        model = st.session_state.model
        sp = st.session_state.sp
        device = st.session_state.device
        download_status = st.session_state.download_status
        error_msg = st.session_state.error_msg
    
    # Sidebar with download status
    show_download_status(download_status)
    
    st.sidebar.markdown("---")
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a Sequence-to-Sequence (Seq2Seq) model with LSTM layers "
        "for text transliteration. The model is trained on character-level representations "
        "using SentencePiece tokenization."
    )
    
    # Manual download section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Manual Download")
    st.sidebar.markdown("""
    If automatic download fails:
    1. Go to [GitHub Repository](https://github.com/barirazaib/seq2seq)
    2. Find the files in the main branch
    3. Click on each file → 'Download' button
    4. Save in this app's folder
    """)
    
    # Check if model loaded successfully
    if model is None:
        st.error(f"""
        ❌ {error_msg}
        
        **Quick Fix Options:**
        
        **Option 1: Retry Download**
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry Automatic Download", type="primary", use_container_width=True):
                st.cache_resource.clear()
                if 'initialized' in st.session_state:
                    del st.session_state['initialized']
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Cache & Retry", use_container_width=True):
                st.cache_resource.clear()
                # Remove existing files
                for file in ['best_seq2seq_joint.pth', 'joint_char.model']:
                    if os.path.exists(file):
                        os.remove(file)
                if 'initialized' in st.session_state:
                    del st.session_state['initialized']
                st.rerun()
        
        st.markdown("""
        **Option 2: Manual Download**
        
        Please manually download these files:
        
        1. **[best_seq2seq_joint.pth](https://github.com/barirazaib/seq2seq/raw/main/best_seq2seq_joint.pth)**
           - Right-click → "Save link as..."
           - Save as `best_seq2seq_joint.pth`
           - Expected size: ~5-50 MB
        
        2. **[joint_char.model](https://github.com/barirazaib/seq2seq/raw/main/joint_char.model)**
           - Right-click → "Save link as..." 
           - Save as `joint_char.model`
        
        Then click the retry button above.
        """)
        return
    
    # Show success message
    st.success(f"✅ Model loaded successfully on **{device}**! You can now start transliterating text.")
    
    # Debug information (can be removed later)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Debug Info")
    st.sidebar.write(f"Model: {'Loaded' if model else 'None'}")
    st.sidebar.write(f"Tokenizer: {'Loaded' if sp else 'None'}")
    st.sidebar.write(f"Device: {device}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Input Text")
        
        # Initialize session state for input text if not exists
        if 'input_text' not in st.session_state:
            st.session_state.input_text = ""
        
        input_text = st.text_area(
            "Enter text to transliterate:",
            value=st.session_state.input_text,
            placeholder="Type your text here...",
            height=150,
            key="input_text_area"
        )
        
        # Update session state
        st.session_state.input_text = input_text
        
        # Quick examples
        st.markdown("**Quick examples:**")
        examples = ["ہم نے ایک خوبصورت باغ دیکھا۔", "آپ کا نام کیا ہے؟", "میں سکول جاتا ہوں"]
        cols = st.columns(3)
        for i, example in enumerate(examples):
            if cols[i].button(example, use_container_width=True, key=f"example_{i}"):
                st.session_state.input_text = example
                st.rerun()
    
    with col2:
        st.subheader("📤 Output")
        
        # Always show the transliterate button when model is loaded
        st.markdown("### Transliteration")
        
        # Show current input text for debugging
        if st.session_state.input_text:
            st.info(f"**Input:** {st.session_state.input_text}")
        
        # The transliterate button - ALWAYS VISIBLE when model is loaded
        if st.button("🚀 Transliterate", 
                    use_container_width=True, 
                    type="primary",
                    key="transliterate_btn_main"):
            
            if not st.session_state.input_text:
                st.warning("⚠️ Please enter some text to transliterate.")
            else:
                with st.spinner("🔄 Transliterating..."):
                    try:
                        # Perform transliteration
                        st.write("🔍 Starting transliteration...")
                        output_text = transliterate_text(model, st.session_state.input_text, sp, device)
                        
                        # Display result
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown("**Transliterated Text:**")
                        st.success(output_text)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Copy to clipboard functionality
                        st.code(output_text, language="text")
                        
                        # Show success message
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error during transliteration: {str(e)}")
                        st.info("💡 Try a different text or check if the model files are complete.")
        
        # Show instruction when no text is entered
        if not st.session_state.input_text:
            st.info("👆 Enter some text in the left panel and click the **Transliterate** button above!")
        
        # Show additional help
        st.markdown("---")
        st.markdown("### 💡 Need help?")
        st.markdown("""
        - Enter text in the left panel
        - Click the **Transliterate** button above
        - View results in this section
        - Try the example buttons for quick testing
        """)
    
    # Additional information
    st.markdown("---")
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("### ℹ️ Model Info")
        st.markdown(f"""
        - **Architecture**: Seq2Seq with LSTM
        - **Encoder**: Bidirectional LSTM  
        - **Device**: {device}
        - **Vocab Size**: {sp.get_piece_size() if sp else 'N/A'}
        - **Model File**: {os.path.getsize('best_seq2seq_joint.pth') // 1024 if os.path.exists('best_seq2seq_joint.pth') else 0} KB
        """)
    
    with col_info2:
        st.markdown("### 📝 Usage Tips")
        st.markdown("""
        - Enter text in the input box (left panel)
        - Click the **Transliterate** button (right panel)
        - View results in the output section
        - Use quick example buttons for testing
        - Results are character-level transliterations
        """)

if __name__ == "__main__":
    main()
