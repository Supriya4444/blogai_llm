import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
import os

# Set your Hugging Face token here (best to set via environment variable)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

## Function to get response from LLaMA 3.2 1B model
def getLLamaResponse(input_text, no_words, blog_style):
    # Load model from Hugging Face Hub
    llm = HuggingFaceHub(
        repo_id="meta-llama/Llama-3.2-1B",
        model_kwargs={"max_new_tokens": 2048, "temperature": 0.7}
    )

    ## Prompt Template
    template = """
    Write a blog for a {blog_style} job profile on the topic "{input_text}"
    within {no_words} words.
    """

    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", "no_words"],
        template=template
    )

    ## Generate the response
    final_prompt = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
    response = llm(final_prompt)
    print(response)
    return response

## Streamlit Frontend
st.set_page_config(page_title="Generate Blogs", page_icon='🤖', layout='centered', initial_sidebar_state='collapsed')
st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

## Creating two more columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button("Generate")

## Final response
if submit:
    st.write(getLLamaResponse(input_text, no_words, blog_style))