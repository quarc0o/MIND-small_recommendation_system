import pandas as pd

# Load the DataFrame
train_df = pd.read_csv('data/training/behaviors.tsv', delimiter='\t', header=None)

# Filter away articles before a certain date
train_df[2] = pd.to_datetime(train_df[2])
train_df = train_df.loc[train_df[2] >= pd.Timestamp('2019-11-14')]

impression_log_col = train_df.iloc[:, -1]

articles_click_count = {}

for impression_data in impression_log_col:
  news_impressions = impression_data.split()
  for single_impression in news_impressions:
    try:
      article_id, is_clicked = single_impression.split("-")
    except:
      continue
    if is_clicked == '1':
      if article_id in articles_click_count:
        articles_click_count[article_id] += 1
      else:
        articles_click_count[article_id] = 1

sorted_articles_clicks = sorted(articles_click_count.items(), key=lambda x: x[1], reverse=True)

with open('models/baseline/most_popular_articles.txt', 'w') as file:
  for article_id, count in sorted_articles_clicks[:10]:
      print(f"Article {article_id} was clicked {count} times.")
      file.write(f"{article_id}\n")