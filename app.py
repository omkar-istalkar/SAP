from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import os

app = Flask(__name__)

# Update paths to reference the templates folder correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOKS_PATH = os.path.join(BASE_DIR, 'templates', 'Books.csv')
RATINGS_PATH = os.path.join(BASE_DIR, 'templates', 'Ratings.csv')

# Load datasets
books = pd.read_csv(BOOKS_PATH, low_memory=False)
ratings = pd.read_csv(RATINGS_PATH)

# Fill missing values and merge datasets
books.fillna("", inplace=True)
ratings_with_books = ratings.merge(books[['ISBN', 'Book-Title', 'Book-Author', 'Image-URL-L']], on='ISBN', how='left')

# Filter books with more than 50 ratings
book_counts = ratings_with_books['Book-Title'].value_counts()
popular_books = book_counts[book_counts >= 50].index
filtered_ratings = ratings_with_books[ratings_with_books['Book-Title'].isin(popular_books)]

# Create user-item matrix
user_item_sparse = filtered_ratings.pivot_table(index='User-ID', columns='Book-Title', values='Book-Rating',aggfunc='mean').fillna(0)
sparse_matrix = csr_matrix(user_item_sparse.values)

# Fit Nearest Neighbors Model
nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(sparse_matrix)


def recommend_books_by_title(book_title, top_n=5):
    book_title = book_title.lower()
    # Check if the book title exists in the dataset
    if book_title not in books['Book-Title'].str.lower().values:
        return []

    # Find the index of the book
    book_idx = books[books['Book-Title'].str.lower() == book_title].index[0]

    # Calculate similarity based on book titles
    distances, indices = nn_model.kneighbors(sparse_matrix[book_idx], n_neighbors=top_n + 1)

    recommended_books = books.iloc[indices.flatten()[1:]][
        ['Book-Title', 'Book-Author', 'Image-URL-L']].drop_duplicates()
    return recommended_books.to_dict(orient='records')


def recommend_books_by_author(author_name, top_n=5):
    author_name = author_name.lower()

    # Filter books by the given author
    author_books = books[books['Book-Author'].str.lower().str.contains(author_name)]

    if author_books.empty:
        return []

    # Pick top N books from the same author
    return author_books[['Book-Title', 'Book-Author', 'Image-URL-L']].head(top_n).to_dict(orient='records')


# ---- Flask Routes ----
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.form.get('query')  # Book title or author name
    search_type = request.form.get('search_type')  # Either 'title' or 'author'

    if search_type == 'title':
        recommendations = recommend_books_by_title(query)
    elif search_type == 'author':
        recommendations = recommend_books_by_author(query)
    else:
        recommendations = []

    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)
