from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI()

# Print current working directory
print("Current Working Directory:", os.getcwd())

# Absolute file paths for local machine (Windows)
csv_path_det_local = r'C:\Users\DELL\Downloads\Netflix_Collaborative_RS\Netflix_Dataset_Det.csv'
csv_path_movie_local = r'C:\Users\DELL\Downloads\Netflix_Collaborative_RS\Netflix_Dataset_Movie.csv'
csv_path_rating_local = r'C:\Users\DELL\Downloads\Netflix_Collaborative_RS\Netflix_Dataset_Rating.csv'

# Absolute file paths for cloud VM instance (Linux/Unix)
csv_path_det_cloud = '/home/ec2-user/Netflix-Collaborative-RS/Netflix_Dataset_Det.csv'
csv_path_movie_cloud = '/home/ec2-user/Netflix-Collaborative-RS/Netflix_Dataset_Movie.csv'
csv_path_rating_cloud = '/home/ec2-user/Netflix-Collaborative-RS/Netflix_Dataset_Rating.csv'

# Choose the appropriate paths based on the environment (local/cloud)
if os.name == 'nt':  # Windows
    csv_path_det = csv_path_det_local
    csv_path_movie = csv_path_movie_local
    csv_path_rating = csv_path_rating_local
else:  # Linux/Unix
    csv_path_det = csv_path_det_cloud
    csv_path_movie = csv_path_movie_cloud
    csv_path_rating = csv_path_rating_cloud

# Load your Netflix dataset into a DataFrame
df = pd.read_csv(csv_path_det, encoding="ISO-8859-1")

# Load your movie and rating data into matrices
df_movie = pd.read_csv(csv_path_movie)
df_rating = pd.read_csv(csv_path_rating)

# Merging them into one DataFrame
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
                return f"The user with {user_id} would rate {movie_id} as {rating}, therefore it is not advisable to recommend the given movie to the given user."
            else:
                return f"The user with {user_id} would rate {movie_id} as {rating}, therefore it is advisable to recommend the given movie to the given user."

@app.get("/")
async def read_root():
    return "Hola!"

@app.post("/calculate_rating/")
async def calculate_rating(user_id: int, movie_id: int, n: int):
    # Call your recommendation function and return the result
    result = calculate_weighted_rating(user_id, movie_id, n)
    return {"recommendation": result}
