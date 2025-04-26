import streamlit as st
from utils.scraper import RestaurantScraper
from utils.config import load_urls
from utils.embedder import Embedder
from utils.retriever import Retriever
import google.generativeai as genai
import json
import os

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-pro')

# Initialize components
embedder = Embedder()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

def load_or_create_embeddings():
    if os.path.exists("data/scraped_data.json"):
        with open("data/scraped_data.json", "r") as f:
            data = json.load(f)
        embeddings = embedder.create_embeddings(data)
        st.session_state.retriever = Retriever(embeddings)
        return True
    return False

def main():
    st.title("🍔 Zomato RAG Restaurant Assistant")
    
    # Initialize embeddings if data exists
    if st.session_state.retriever is None:
        load_or_create_embeddings()
    
    # Sidebar for data management
    with st.sidebar:
        st.header("Data Management")
        
        # URL upload
        uploaded_urls = st.file_uploader("Upload URLs file", type=["txt"])
        if uploaded_urls:
            with open("data/website_urls.txt", "wb") as f:
                f.write(uploaded_urls.getvalue())
        
        # Scrape button
        if st.button("Scrape Websites & Build RAG"):
            with st.spinner("Building knowledge base..."):
                urls = load_urls()
                scraper = RestaurantScraper(urls)
                data = scraper.scrape()
                
                # Create and store embeddings
                embeddings = embedder.create_embeddings(data)
                st.session_state.retriever = Retriever(embeddings)
                
                # Save data
                with open("data/scraped_data.json", "w") as f:
                    json.dump(data, f)
                
                st.success("RAG system ready!")

    # Chat interface
    st.header("Ask About Restaurants")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask about restaurants..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.spinner("Searching knowledge base..."):
            if st.session_state.retriever:
                context = st.session_state.retriever.retrieve(prompt, embedder)
                response = generate_response(prompt, context)
            else:
                response = "Please load restaurant data first (click 'Scrape Websites' in sidebar)"
        
        # Add assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display response
        with st.chat_message("assistant"):
            st.markdown(response)
            if st.session_state.retriever:
                with st.expander("Sources"):
                    for src in context:
                        st.write(f"**{src['name']}**")
                        st.caption(src.get('contact', 'No contact info'))

def generate_response(prompt, context):
    """Enhanced RAG response generation"""
    context_str = "\n\n".join([
        f"""Restaurant: {r['name']}
Menu Items: {', '.join([item['name'] for item in r.get('menu', [])])}
Features: {r.get('features', 'N/A')}
Contact: {r.get('contact', 'N/A')}"""
        for r in context
    ])
    
    prompt_template = f"""
    You are a restaurant information assistant. Use the following context to answer the question.
    
    Context:
    {context_str}
    
    Question: {prompt}
    
    Guidelines:
    - Be concise but helpful
    - If you don't know, say so
    - Mention restaurant names when relevant
    - Format prices clearly
    """
    
    response = model.generate_content(prompt_template)
    return response.text

if __name__ == "__main__":
    main()