# app.py
import streamlit as st
import torch
import sys
import os
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
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def download_model_files():
    """Function to handle model file downloads - users need to download manually"""
    st.sidebar.markdown("### 📁 Download Model Files")
    st.sidebar.markdown("""
    You need to download these files and place them in the same directory as this app:
    
    **Required Files:**
    - [best_seq2seq_joint.pth](https://github.com/barirazaib/seq2seq/blob/main/best_seq2seq_joint.pth)
    - [joint_char.model](https://github.com/barirazaib/seq2seq/blob/main/joint_char.model)
    
    ⚠️ **Right-click → 'Save link as...'** to download each file.
    """)

def check_model_files():
    """Check if required model files exist"""
    required_files = ['best_seq2seq_joint.pth', 'joint_char.model']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

def load_model_with_progress():
    """Load model with progress indicator"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Loading model...")
    progress_bar.progress(30)
    
    try:
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        status_text.text(f"Using device: {device}")
        progress_bar.progress(50)
        
        # Load model
        model, sp = load_model(
            model_path="best_seq2seq_joint.pth",
            sp_model_path="joint_char.model",
            device=device
        )
        progress_bar.progress(80)
        
        status_text.text("Model loaded successfully!")
        progress_bar.progress(100)
        
        return model, sp, device
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🔤 Text Transliteration System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a Sequence-to-Sequence (Seq2Seq) model with LSTM layers "
        "for text transliteration. The model is trained on character-level representations "
        "using SentencePiece tokenization."
    )
    
    # Model file download instructions
    download_model_files()
    
    # Check for model files
    missing_files = check_model_files()
    
    if missing_files:
        st.error(f"❌ Missing required model files: {', '.join(missing_files)}")
        st.markdown("""
        ### Please download the required files:
        1. Go to the GitHub repository links in the sidebar
        2. Download both files
        3. Place them in the same directory as this app
        """)
        return
    
    # Load model (cached to avoid reloading on every interaction)
    @st.cache_resource(show_spinner=False)
    def cached_load_model():
        return load_model_with_progress()
    
    model, sp, device = cached_load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check the model files.")
        return
    
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
        
        if input_method == "Text Input":
            input_text = st.text_area(
                "Enter text to transliterate:",
                placeholder="Type your text here...",
                height=150
            )
        else:
            example_options = {
                "Hello world": "Hello world",
                "How are you?": "How are you?",
                "Machine learning": "Machine learning",
                "Natural language processing": "Natural language processing"
            }
            selected_example = st.selectbox(
                "Choose an example:",
                list(example_options.keys())
            )
            input_text = example_options[selected_example]
            st.text_area("Selected example:", value=input_text, height=100, disabled=True)
    
    with col2:
        st.subheader("📤 Output")
        
        if input_text and st.button("🚀 Transliterate", use_container_width=True):
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
        - **Decoder**: LSTM with Attention
        - **Tokenization**: SentencePiece
        """)
    
    with col_info2:
        st.markdown("### ⚙️ Technical Details")
        st.markdown("""
        - **Embedding Dim**: 256
        - **Hidden Dim**: 256
        - **Layers**: 2
        - **Dropout**: 0.3
        - **Device**: {}
        """.format(device))
    
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
