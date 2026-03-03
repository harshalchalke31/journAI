import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import  OutputParserException
import pandas as pd
import chromadb
import uuid
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY', None)
if groq_api_key is None:
    st.error("GROQ_API_KEY is not set. Please set it in your environment variables.")
    st.stop()  # Stop execution if API key is missing

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")


class LMChain:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model="mixtral-8x7b-32768",
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            Extract job postings in JSON format containing keys: `role`, `experience`, `skills`, and `description`.
            Only return valid JSON.
            """
        )

        chain_extract = prompt_extract | self.llm
        try:
            res = chain_extract.invoke(cleaned_text)  # Fix: Remove incorrect `input=`
            json_parser = JsonOutputParser()
            res = json_parser.parse(res)
        except OutputParserException:
            raise OutputParserException("Content too long. Unable to parse Jobs.")
        except Exception as e:
            st.error(f"Error extracting jobs: {e}")
            return []

        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            I am Harshal Chalke, an AI and ML professional. My portfolio: {link_list}
            Generate an email applying for this job.
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        try:
            res = chain_email.invoke({"job_description": str(job), "link_list": links})
            return res.content if hasattr(res, 'content') else "Error: No response from LLM."
        except Exception as e:
            st.error(f"Error generating email: {e}")
            return "Error in email generation."


class Portfolio:
    def __init__(self, filepath=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'csv_data', 'my_portfolio.csv'))):
        self.filepath = filepath
        try:
            self.data = pd.read_csv(filepath)
        except FileNotFoundError:
            st.error(f"CSV file not found at {filepath}")
            st.stop()

        required_columns = {"Techstack", "Links"}
        if not required_columns.issubset(self.data.columns):
            st.error(f"CSV is missing required columns: {required_columns}")
            st.stop()

        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection('Portfolio')

    def load_portfolio(self):
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(documents=row["Techstack"],
                                    metadatas={"links": row["Links"]},
                                    ids=[str(uuid.uuid4())])

    def query_links(self, skills):
        result = self.collection.query(query_texts=skills, n_results=2)
        return [meta['links'] for sublist in result.get('metadatas', []) for meta in sublist]


def clean_text(text):
    return re.sub(r'<[^>]*?>|http[s]?://\S+|[^a-zA-Z0-9 ]', ' ', text).strip()


def create_app5(llm, portfolio):
    st.title("Cold Email Generator:")
    url_input = st.text_input(label="Enter a URL of the Job posting:")
    submit = st.button("Submit")

    if submit:
        try:
            with st.spinner("Reading the website..."):
                loader = WebBaseLoader([url_input])
                website_content = loader.load()

                if not website_content:
                    st.error("Error: No content found on the website.")
                    return

                data = clean_text(website_content.pop().page_content)
            st.success("Website content extracted!")

            with st.spinner("Loading portfolio data..."):
                portfolio.load_portfolio()
            st.success("Portfolio loaded successfully!")

            with st.spinner("Extracting job information..."):
                jobs = llm.extract_jobs(data)

                if not jobs:
                    st.error("No jobs were extracted. The webpage may not have structured job listings.")
                    return
            st.success("Job information extracted!")

            for job in jobs:
                with st.spinner("Fetching relevant portfolio links..."):
                    skills = job.get('skills', [])
                    links = portfolio.query_links(skills)

                with st.spinner("Generating email..."):
                    email = llm.write_mail(job, links)

                st.success("Email generated!")
                st.code(email, language='markdown')

        except Exception as e:
            st.error(f"An Error Occurred: {e}")
            st.write(e)  # Print error for debugging


if __name__ == "__main__":
    chain = LMChain()
    portfolio = Portfolio()
    create_app5(chain, portfolio)



# import streamlit as st
# from langchain_community.document_loaders import WebBaseLoader
# # from utils.app5_util import LMChain,Portfolio,clean_text

# import os
# from langchain_groq import ChatGroq
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.exceptions import OutputParserException
# import pandas as pd
# import chromadb
# import uuid
# import re
# from dotenv import load_dotenv
# load_dotenv()
# groq_api_key = os.getenv('GROQ_API_KEY',None)

# class LMChain:
#     def __init__(self):
#         self.llm = ChatGroq(
#             api_key=groq_api_key,
#             model="mixtral-8x7b-32768",
#         )

#     def extract_jobs(self,cleaned_text):
#         prompt_extract = PromptTemplate.from_template(
#             template="""
#             ### SCRAPED TEXT FROM WEBSITE:
#             {page_data}
#             ### INSTRUCTION:
#             The scraped text is from the career's page of a website.
#             Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
#             Only return the valid JSON.
#             ### VALID JSON (NO PREAMBLE):
#             """
#         )

#         chain_extract = prompt_extract | self.llm
#         res = chain_extract.invoke(input={"page_data":cleaned_text})
#         try:
#             json_parser = JsonOutputParser()
#             res = json_parser.parse(res.content)
#         except:
#             raise OutputParserException("Content too long. Unable to parse Jobs.")
#         return res if isinstance(res,list) else [res]

#     def write_mail(self, job, links):
#         prompt_email = PromptTemplate.from_template(
#             """
#             ### JOB DESCRIPTION:
#             {job_description}

#             ### INSTRUCTION:
#             I am Harshal Chalke, a passionate AI and Machine Learning professional pursuing a Master’s in Artificial Intelligence
#             at Rochester Institute of Technology (RIT, with expertise in transformers, diffusion models, and machine learning algorithms.
#             I have hands-on experience in AI-driven applications, having worked as a Data Science Intern at United Mentor, where I developed 
#             a Conversational PDF Assistant using Mistral LLM and Retrieval-Augmented Generation (RAG) for intelligent document management. 
#             Additionally, as an Android Development Intern at Being Digital, I contributed to building a mobile application aligned with the 
#             company’s website. My projects include CareVision AI, where I implemented UNet models for medical image segmentation with 85%+ 
#             Dice scores, and FortiFace AI, a Siamese Neural Network-based facial recognition system enhanced with GANs for improved accuracy 
#             in low-light conditions.
#             Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
#             Remember you are Harshal, MS in AI student at RIT. 
#             Do not provide a preamble.
#             ### EMAIL (NO PREAMBLE):

#             """
#         )
#         chain_email = prompt_email | self.llm
#         res = chain_email.invoke({"job_description": str(job), "link_list": links})
#         return res.content

# class Portfolio:
#     def __init__(self,filepath=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'csv_data', 'my_portfolio.csv'))):
#         self.filepath=filepath
#         self.data = pd.read_csv(filepath)
#         self.chroma_client = chromadb.Client()
#         self.collection = self.chroma_client.get_or_create_collection('Portfolio')

#     def load_portfolio(self):
#         if not self.collection.count():
#             for _, row in self.data.iterrows():
#                 self.collection.add(documents=row["Techstack"],
#                                     metadatas={"links": row["Links"]},
#                                     ids=[str(uuid.uuid4())])
#     def query_links(self,skills):
#         return self.collection.query(query_texts=skills,n_results=2).get('metadatas',[])
    
# def clean_text(text):
#     # Remove HTML tags
#     text = re.sub(r'<[^>]*?>', '', text)
#     # Remove URLs
#     text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
#     # Remove special characters
#     text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
#     # Replace multiple spaces with a single space
#     text = re.sub(r'\s{2,}', ' ', text)
#     # Trim leading and trailing whitespace
#     text = text.strip()
#     # Remove extra whitespace
#     text = ' '.join(text.split())
#     return text




# def create_app5(llm, portfolio, clean_text):
#     st.title("Cold Email Generator:")
#     url_input = st.text_input(label="Enter a URL of the Job posting:")
#     submit = st.button("Submit")

#     if submit:
#         try:
#             with st.spinner("Reading the website ..."):
#                 loader = WebBaseLoader([url_input])
#                 data = clean_text(loader.load().pop().page_content)
#             st.success("Website content extracted!")

#             with st.spinner("Loading portfolio data..."):
#                 portfolio.load_portfolio()
#             st.success("Portfolio loaded successfully!")

#             with st.spinner("Extracting job information..."):
#                 jobs = llm.extract_jobs(data)
#             st.success("Job information extracted!")

#             for job in jobs:
#                 with st.spinner("Fetching relevant portfolio links..."):
#                     skills = job.get('skills', [])
#                     links = portfolio.query_links(skills)

#                 with st.spinner("Generating email..."):
#                     email = llm.write_mail(job, links)
#                 st.success("Email generated!")

#                 st.code(email, language='markdown')

#         except Exception as e:
#             st.error(f"An Error Occurred: {e}")


# if __name__ == "__main__":
#     chain = LMChain()
#     portfolio = Portfolio()
#     st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
#     create_app5(chain, portfolio, clean_text)