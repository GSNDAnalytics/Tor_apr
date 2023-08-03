import streamlit as st
import pandas as pd
import numpy as np
import scipy
from scipy import sparse
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import base64



image_file = Image.open('food-outline-icon-on-blue-background-breakfast-background-seamless-pattern-for-printing-wallpaper-decoration-illustration-free-vector.jpeg')


#st.markdown(page_bg_img, unsafe_allow_html=True)

#def image_local(image_file):
    #with open(image_file, "rb") as image_file:
        #encoded_string = base64.b64encode(image_file.read())
    #st.markdown(
    #f"""
    #<style>
    #.stApp {{
    #    background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
    #    background-size: cover
    #}}
    #</style>
    #""",
    #unsafe_allow_html=True
    #)
#image_local('photo.png') 

def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
    st.markdown(
         f"""
         <style>
         section.css-vk3wp9.e1akgbir11{{
             background: url(https://static.vecteezy.com/system/resources/previews/004/930/845/non_2x/food-outline-icon-on-blue-background-breakfast-background-seamless-pattern-for-printing-wallpaper-decoration-illustration-free-vector.jpg);
    background-size: cover;
    background-blend-mode: lighten;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_bg_hack_url()

def sidebar_bg(side_bg):

   image_file = Image.open('food-outline-icon-on-blue-background-breakfast-background-seamless-pattern-for-printing-wallpaper-decoration-illustration-free-vector.jpeg')

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{image_file};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )

@st.cache_data
def load_model():
    sample = pd.read_csv(".DS_Store")
    
    vectorizer = TfidfVectorizer(stop_words = "english", min_df=2)
    sample['name'] = sample['name'].fillna("")

    TF_IDF_matrix = vectorizer.fit_transform(sample['name'])

    similarity = cosine_similarity(TF_IDF_matrix, dense_output=False)

    

    return sample, similarity


def content_recommender(title, similarities, vote_threshold=10, top_n=10) :
    
    # Get the movie by the title
    
    matches = sample[sample['name'].str.contains(title)]
    if len(matches) == 0:
        st.write("No matches for this one.")
        return pd.DataFrame()

    recipe_index = matches.index[0]
    #recipe_index2 = drop_duplicates(recipe_index, subset=None, keep="first", inplace=False)

        
    st.write("We are matching this recipe for you:")
    st.write(sample.loc[recipe_index, "name"])

   # st.write(np.array(similarities[recipe_index, :].todense()).squeeze().shape)
    
    # Create a dataframe with the movie titles
    sim_df = pd.DataFrame(
        {'Recipe': sample['name'], 
         'Similarity Score': np.array(similarities[recipe_index, :].todense()).squeeze(),
         'Review Count': sample['count']
        })
    
    # Get the top 10 movies with > 10 votes
    top_recipes = sim_df[sim_df['Review Count'] > vote_threshold].sort_values(by='Similarity Score', ascending=False).head(top_n)
    
    return top_recipes



st.title('Recipe Recommender')

sample, similarity = load_model()


name = st.text_input(
    'Which recipe did you make recently?',)

with st.sidebar:
    threshold = st.slider("Novelty threshold", 50, 1000, step=50)
    top = st.slider("Top N", 5, 50, step=5)
    #st.image(image_file)
    #image_local(image_file)
    #sidebar_bg(image_file)
 


df = content_recommender(name, similarity, vote_threshold=threshold, top_n=top)

if len(df) > 0:
    st.header("Recommendations:")
    st.write(df)
