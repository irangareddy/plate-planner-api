import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load your dataset
df = pd.read_csv('/src/data/RecipeNLG_dataset.csv')

# Combine all directions into a big string
all_directions = ""

for directions in tqdm(df['directions']):
    if pd.isna(directions):
        continue
    if isinstance(directions, list):
        all_directions += " ".join(directions).lower() + " "
    else:
        all_directions += str(directions).lower() + " "

# Generate word cloud
wordcloud = WordCloud(width=1600, height=800, background_color='white', max_words=200).generate(all_directions)

# Display the generated image
plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Common Words in Recipe Directions", fontsize=20)
plt.show()
