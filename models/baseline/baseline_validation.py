import pandas as pd

#load df
validation_df = pd.read_csv('data/validation/behaviors.tsv', delimiter='\t', header=None)

impression_log_col = validation_df.iloc[:, -1]

# Load the list of most popular articles
with open('models/baseline/most_popular_articles.txt', 'r') as file:
    popular_articles = [line.strip() for line in file.readlines()]

# Hit rate
total_articles_shown = 0
total_articles_clicked = 0

for user_hits in impression_log_col:
    articles = user_hits.split()  
    for article in articles:
        article_id, clicked = article.split('-')
        total_articles_shown += 1
        if clicked == '1':
            if article_id in popular_articles:
                total_articles_clicked += 1

hit_rate = total_articles_clicked / total_articles_shown if total_articles_shown else 0
print(f"Hit Rate: {hit_rate:.4%}")