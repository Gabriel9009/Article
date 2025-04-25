import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import streamlit as st

@st.cache_data  # Cache the data loading for better performance
def load_articles():
    """Load articles data with robust path handling and error checking"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct potential paths to the CSV file
        possible_paths = [
            os.path.join(script_dir, "articles.csv"),  # Same directory as script
            os.path.join(script_dir, "data", "articles.csv"),  # In a data subdirectory
            os.path.join(os.getcwd(), "articles.csv"),  # Current working directory
            "/mount/src/article/Article/articles.csv"  # Absolute path you mentioned
        ]
        
        # Try each possible path
        for path in possible_paths:
            try:
                if not os.path.exists(path):
                    continue
                
                df = pd.read_csv(path)
                
                # Validate required columns exist
                required_columns = {'Title', 'Article'}
                if not required_columns.issubset(df.columns):
                    raise ValueError(f"CSV missing required columns. Needs: {required_columns}")
                
                st.success(f"Successfully loaded articles from: {path}")
                return df
                
            except pd.errors.EmptyDataError:
                raise ValueError(f"File is empty: {path}")
            except pd.errors.ParserError:
                raise ValueError(f"Error parsing CSV file: {path}")
        
        # If we get here, no path worked
        available_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]
        raise FileNotFoundError(
            f"Could not find articles.csv in:\n"
            f"Script directory: {script_dir}\n"
            f"Available CSV files: {available_files}\n"
            f"Tried paths: {possible_paths}"
        )
        
    except Exception as e:
        st.error(f"Error loading articles: {str(e)}")
        return None

# Load articles with error handling
df = load_articles()
if df is None:
    st.stop()

# Process articles
try:
    articles = df["Article"].tolist()
    
    # Vectorize articles
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(articles)
    
    # Similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
except Exception as e:
    st.error(f"Error processing articles: {str(e)}")
    st.stop()

def recommend_articles(title, top_n=5):
    """Get article recommendations with input validation"""
    try:
        if not isinstance(title, str):
            return ["Invalid input: title must be a string"]
            
        if title.strip() == "":
            return ["Please enter an article title"]
            
        if title not in df['Title'].values:
            return [f"Article not found: '{title}'"]
        
        idx = df[df['Title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]

        return [df.iloc[i[0]]['Title'] for i in sim_scores]
        
    except Exception as e:
        st.error(f"Recommendation error: {str(e)}")
        return ["Error generating recommendations"]

# Streamlit UI
# st.set_page_config(layout="wide")

# Sidebar with article selection
with st.sidebar:
    st.header("📚 Available Articles")
    selected_article = st.selectbox(
        "Choose an article to get recommendations:",
        df['Title'].tolist(),
        index=0,
        help="Select an article from the list to see similar recommendations"
    )
    
    st.markdown("---")
    st.markdown("### How to use:")
    st.markdown("1. Select an article from the list")
    st.markdown("2. View recommendations on the right")
    st.markdown("3. Or search for any article in the search box")

# Main content area
st.title("📖 Article Recommender System")
st.markdown("Discover articles similar to the one you're reading")

# Search functionality
search_query = st.text_input(
    "Or search for an article:",
    placeholder="Type an article title..."
)

# Determine which article to use (sidebar selection or search)
article_to_recommend = search_query if search_query else selected_article

# Get and display recommendations
recommendations = recommend_articles(article_to_recommend)

if recommendations and not recommendations[0].startswith(("Article not found", "Invalid input", "Please enter")):
    st.subheader(f"Recommended articles similar to: '{article_to_recommend}'")
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
elif recommendations:
    st.warning(recommendations[0])
else:
    st.info("No recommendations available")
