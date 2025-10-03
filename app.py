# app_simple.py - ULTRA SIMPLE VERSION
import streamlit as st
import torch
import os
from model import load_model, transliterate_text

st.set_page_config(page_title="Transliteration App", layout="wide")

# Initialize session state
if 'current_text' not in st.session_state:
    st.session_state.current_text = ""
if 'output' not in st.session_state:
    st.session_state.output = ""

st.title("🔤 Text Transliteration")

# Simple examples
examples = [
    "ہم نے ایک خوبصورت باغ دیکھا۔",
    "آپ کا نام کیا ہے؟", 
    "میں سکول جاتا ہوں"
]

# Load model (you'll need to handle this part)
model, sp, device = None, None, "cpu"
# TODO: Add your model loading code here

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    
    # Display example buttons
    st.write("**Click an example:**")
    for i, example in enumerate(examples):
        if st.button(example, key=f"btn_{i}", use_container_width=True):
            st.session_state.current_text = example
            st.rerun()
    
    # Text input
    user_text = st.text_area(
        "Or type your own text:",
        value=st.session_state.current_text,
        height=100
    )
    
    # Update session state
    st.session_state.current_text = user_text
    
    # Transliterate button
    if st.button("🚀 TRANSLITERATE", type="primary", use_container_width=True):
        if st.session_state.current_text.strip():
            # Perform transliteration
            try:
                # TODO: Call your transliterate function here
                # output = transliterate_text(model, st.session_state.current_text, sp, device)
                output = f"Transliterated: {st.session_state.current_text}"  # Placeholder
                st.session_state.output = output
            except Exception as e:
                st.session_state.output = f"Error: {str(e)}"
        else:
            st.session_state.output = "Please enter some text first."

with col2:
    st.subheader("Output")
    
    if st.session_state.output:
        st.success(st.session_state.output)
        st.code(st.session_state.output)
    else:
        st.info("Enter text and click TRANSLITERATE to see results")

    # Debug info
    with st.expander("Debug Info"):
        st.write(f"Current text: {st.session_state.current_text}")
        st.write(f"Output: {st.session_state.output}")
