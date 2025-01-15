import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import ctransformers

## Core function: llama

def llama_llm(topic_name, content_type, word_count,content_style,description):
    
    # model
    llm = ctransformers(model = 'C:\Projects\python\journAI\models\llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type = 'llama',
                        config = {'max_new_tokens':100,
                                'temperature':0.1})
    ## prompt template
    template="""Write a {content_type} for the topic {topic_name}. 
    It is being written for {content_style}, for the description {description}. 
    Make sure you fit it under the word limit of {word_count} words."""

    prompt = PromptTemplate(input_variables=['content_type','topic_name','content_style','description','word_count'],
                            template=template)

    # generate response
    response = llm(prompt.format(content_type = content_type,
                    topic_name = topic_name,
                    content_style=content_style,
                    description=description,
                    word_count=word_count))
    return response






## Application Design

st.set_page_config(page_title='Generate Blogs/Journal',
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs/Write a Journal 🤖")

topic_name = st.text_input("Choose your topic")

col1,col2, col3 = st.columns(spec=[5,5,5],gap='medium',vertical_alignment='center',border=True)

with col1:
    content_type = st.selectbox(label='Choose content type',
                                options=('Blog post','Personal Journal Entry'),
                                index=0)
with col2:
    word_count = st.number_input(label='Select word count')

with col3:
    if content_type== 'Blog post':
        content_style = st.selectbox(label='Writing the blog for',
                                        options=('AI engineer','Researcher','Common people'),
                                        index=0)
    else:
        content_style = st.selectbox(label='Writing the blog for',
                                        options=('Self'),
                                        index=0)

description = st.text_input(label='Elaborate on your topic',max_chars=1000)
submit = st.button('Generate ->')

## Final response
if submit:
    st.write_stream(llama_llm(topic_name=topic_name,
                                content_type=content_type,
                                content_style=content_style,
                                word_count=word_count,
                                description=description))