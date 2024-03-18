import pandas as pd

#load df
validation_df = pd.read_csv('data/validation/behaviors.tsv', delimiter='\t', header=None)

impression_log_col = validation_df.iloc[:, -1]

# Load the list of most popular articles
with open('output/most_popular_articles.txt', 'r') as file:
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

# Precision and recall
true_positives = 0
false_positives = 0
false_negatives = 0

# Analyze the impression logs
for user_hits in impression_log_col:
    articles = user_hits.split()
    for article in articles:
        article_id, clicked = article.split('-')
        if article_id in popular_articles:
            if clicked == '1':
                true_positives += 1  
            else:
                false_positives += 1 
        else:
            if clicked == '1':
                false_negatives += 1 

# Calculate precision and recall
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0

print(f"Precision: {precision:.4%}")
print(f"Recall: {recall:.4%}")