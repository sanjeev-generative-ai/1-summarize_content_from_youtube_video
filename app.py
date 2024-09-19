import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader


st.set_page_config(page_title="LangChain: Summarize Text from Youtube")
st.title("LangChain: Summarize Text from Youtube Video")
st.subheader("Summarize URL")

api_key = st.text_input("Enter the Groq API Key", value="", type="password")

if api_key:
    llm = ChatGroq(model="Gemma-7b-It", groq_api_key=api_key)

    URL = st.text_input("URL", label_visibility="collapsed")

    prompt_template = """
    Provide a summary of the following content in 300 words:
    Content:{text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    if st.button("Summarize the Content from this Youtube Video"):
        with st.spinner("Waiting ..."):
            loader = YoutubeLoader.from_youtube_url(URL, add_video_info=True)
            docs=loader.load()
            chain=load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
            output_summary=chain.run(docs)
            st.success(output_summary)
            