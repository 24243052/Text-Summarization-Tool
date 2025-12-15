import streamlit as st
from transformers import pipeline


st.set_page_config(page_title="Text Summarizer Tool", page_icon="📝")
st.title("📝 AI Text Summarizer")
st.markdown("Paste your long text below, and this AI tool will summarize it for you using Natural Language Processing.")


with st.sidebar:
    st.header("Settings")
    max_L = st.slider("Max Summary Length (words)", min_value=30, max_value=500, value=130, step=10)
    min_L = st.slider("Min Summary Length (words)", min_value=10, max_value=100, value=30, step=10)


text_input = st.text_area("Enter your text here:", height=300)


if st.button("Summarize"):
    if not text_input:
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("AI is thinking... (This might take a moment)"):
            try:
                
                summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
                
                
                input_text = text_input[:1024]  
                
                summary = summarizer(input_text, max_length=max_L, min_length=min_L, do_sample=False)
                
               
                st.success("Summary Generated!")
                st.subheader("Your Summary:")
                st.write(summary[0]['summary_text'])
                
            except Exception as e:
                st.error(f"An error occurred: {e}")


st.markdown("---")
st.caption("Built with Python, Transformers, and Streamlit.")
