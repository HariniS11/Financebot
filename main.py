import os
import streamlit as st
import pickle
import time
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

st.title('🤖Articles Insights🔍')
st.sidebar.title('News Articles URLs')

main_placeholder = st.empty()
llm = OpenAI(openai_api_key = api_key, temperature = 0.9 , max_tokens = 500)

urls_list = []
for i in range(2):
    url = st.sidebar.text_input(f'URL {i+1}')
    urls_list.append(url)

search_url_button = st.sidebar.button('Process the URLs')    

if search_url_button:
    loader = UnstructuredURLLoader(urls = urls_list)
    main_placeholder.text("Data Loading...Started...✅✅✅")

    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
       separators=['\n\n', '\n', '.', ','],
       chunk_size=1000)  # chunk_overlap = 200 default
    
    main_placeholder.text("Data Splitting...Started...✅✅✅")

    docs = text_splitter.split_documents(data)
    
    embedding = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs,embedding)
    main_placeholder.text("Embedding Data to vectors started...✅✅✅")
    time.sleep(3)


    file_path = "faiss_vector_database.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input('Ask your Question')        
if query:
    if os.path.exists(file_path):
        with open(file_path,'rb') as f:
            vectorstore = pickle.load(f)
            chain = RecursiveCharacterTextSplitter.from_llm(llm = llm, retriever = vectorstore.as_retriever())
            result = chain({'question':query}, return_only_outputs = True)
            
            st.subheader('Answer')
            st.write(result['answer'])

            sources = result.get("sources", "")
            if sources:
                st.subheader('Sources')
                sources_list = sources.split('\n')
                for source in sources_list:
                    st.write(source)
