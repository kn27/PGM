import numpy as np 
import os
import json

def read():
    folder = '/home/keane/Downloads/hw2-datasets/movielens'
    tags_file = 'tags.csv'
    links_file = 'links.csv'
    ratings_file = 'ratings.csv'
    movies_file = 'movie.csv'
    ratings_matrix_file = 'ratings_matrix.txt'
    movie_map_file = 'movie_map.json'
    ratings = np.zeros((610, 9742))
    if os.path.exists(os.path.join(folder, ratings_matrix_file)):
        ratings = np.loadtxt(os.path.join(folder, ratings_matrix_file))
        with open(os.path.join(folder, movie_map_file)) as file:
            mapping = json.load(file)
    else:
        movie_map = {}
        i = 0 
        count = 0
        with open(os.path.join(folder, ratings_file), 'r') as file:
            while True:
                line = file.readline()
                print(count)
                count +=1
                if not line:
                    break
                elif not line.startswith('user'):
                    try:
                        userId, movieId, rating, _ = line.split(',')
                        movieId, userId, rating =  int(movieId), int(userId), float(rating)
                        assert(rating in np.arange(0.5,5.5,0.5))
                        if movieId not in movie_map:
                            movie_map[movieId] = i
                            i += 1
                        ratings[userId-1, movie_map[movieId]] = rating
                    except:
                        pass
        np.savetxt(os.path.join(folder, ratings_matrix_file), ratings)
        with open(os.path.join(folder, movie_map_file), 'w') as file:
            json.dump(movie_map, file)
    return ratings

if __name__ == "__main__":
    rating = read()