import io
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

app = FastAPI()

# Load your Netflix dataset into a DataFrame
df = pd.read_csv("Netflix_Dataset_Det.csv", encoding="ISO-8859-1")  # or encoding="utf-16"

# Load your movie and rating data into matrices
df_movie = pd.read_csv("Netflix_Dataset_Movie.csv")
df_rating = pd.read_csv("Netflix_Dataset_Rating.csv")

# Merging them into ine DataFrame
merged_df = pd.merge(df_rating, df_movie, on='Movie_ID')

# Creating a utility matrix for the recommender
utility_matrix = merged_df.pivot_table(index='User_ID', columns='Movie_ID', values='Rating')

# Imputing the missing values by 0 for the arithmetic calculations
matrix = utility_matrix.fillna(0)

# Similarity calculator (helper function)
# It returns pairwise cosine similarity scores 
def calculate_movie_similarity(movie_id):
    if movie_id not in matrix.columns:
        return f"Movie ID {movie_id} not found in the utility matrix."
    
    movie_column = matrix[movie_id]
    similarity_scores = cosine_similarity([movie_column.values], matrix.T.values)
    similarity_scores = similarity_scores[0]
    
    similarity_dict = {}
    for i, score in enumerate(similarity_scores):
        if i == movie_column.name or score == 0:
            continue
        similarity_dict[matrix.columns[i]] = round(score, 6)
    
    return similarity_dict

# Your recommendation code function
def calculate_weighted_rating(user_id, movie_id, n):
    if movie_id not in matrix.columns:
        return f"Movie ID {movie_id} not found in the utility matrix."

    if user_id not in matrix.index:
        return f"User ID {user_id} not found in the utility matrix."
    
    rating = matrix.loc[user_id, movie_id]
    
    if rating == 0:
        similarity_scores = calculate_movie_similarity(movie_id)
        if not similarity_scores:
            return "Unable to calculate similarity scores for the movie."
        
        top_movies = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        
        p_values = [score for _, score in top_movies]
        a_values = [movie for movie, _ in top_movies]
        
        valid_movies = all(movie in matrix.columns for movie in a_values)
        if valid_movies:
            r = sum(p * matrix.loc[user_id, a] for p, a in zip(p_values, a_values)) / sum(p_values)
            
            matrix.loc[user_id, movie_id] = r
            return r
        else:
            return "Unable to calculate weighted rating due to missing movie data."
    
    else:
        if type(rating) != str:
            if rating < 2.5:
                return f"The user with {user_id} would rate {movie_id} as {rating} \nTherefore it is not advisable to recommend the given movie to the given user."
            else:
                return f"The user with {user_id} would rate {movie_id} as {rating} \nTherefore it is advisable to recommend the given movie to the given user."

@app.get("/")
async def read_root():
    return "Hola!"

@app.post("/Rating_Recommender/")
async def calculate_rating(user_id: int, movie_id: int, n: int):
    # Call your recommendation function and return the result
    result = calculate_weighted_rating(user_id, movie_id, n)
    if type(result) != str:
        if result< 2.5:
            return f"The user with {user_id} would rate {movie_id} as {result}, therefore it is not advisable to recommend the given movie to the given user"
        if result>= 2.5:
            return f"The user with {user_id} would rate {movie_id} as {result}, therefore it is advisable to recommend the given movie to the given user"
    else:
        return result

@app.get("/eda/")
async def perform_eda():

    # Rename the 'Series or Movie' column
    df.rename(columns={'Series or Movie': 'Type'}, inplace=True)

    # Create a pie chart for Movies vs. Series proportion
    color = ['yellow', 'blue']
    label = ['Series', 'Movies']
    sizes = [df[df['Type'] == 'Series'].size, df[df['Type'] == 'Movie'].size]
    explode = (0.1, 0)

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=label, colors=color, autopct='%2.2f%%')
    ax.axis('equal')

    # Convert the plot to image bytes
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format="png", bbox_inches="tight")
    plt.close()

    # Calculate the language popularity
    language_counts = df['Languages'].str.split(',').explode().value_counts()

    # Select the top 10 most popular languages and reset the index
    top_languages = language_counts.head(10).reset_index()

    # Create a bar plot for language popularity
    language_fig = px.bar(top_languages, y='index', x='Languages', orientation='h',
                          title='Most Popular Languages in Netflix Content', color='Languages',
                          color_continuous_scale='Cividis')
    language_fig.update_layout(yaxis={'categoryorder': 'total ascending'})

    # Convert the language plot to image bytes
    language_img_bytes = io.BytesIO()
    language_fig.write_image(language_img_bytes, format="png")

    # Sort the Netflix dataset by IMDb Score in descending order and select the top 10 rows
    top_10_imdb = df.sort_values(by='IMDb Score', ascending=False).head(10)

    # Create a horizontal bar chart for top IMDb titles
    imdb_fig = px.bar(top_10_imdb, y='Title', x='IMDb Score', orientation='h',
                      title='Top 10 Netflix Titles by IMDb Score', color='IMDb Score',
                      color_continuous_scale='Cividis')
    imdb_fig.update_yaxes(categoryorder='total ascending')

    # Convert the IMDb plot to image bytes
    imdb_img_bytes = io.BytesIO()
    imdb_fig.write_image(imdb_img_bytes, format="png")

    # Return all the plots and information as a StreamingResponse
    return StreamingResponse(
        io.BytesIO(img_bytes.getvalue() + language_img_bytes.getvalue() + imdb_img_bytes.getvalue()),
        media_type="image/png"
    )
