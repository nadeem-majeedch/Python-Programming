"""Generate synthetic product review dataset for sentiment analysis."""

import numpy as np
import pandas as pd
import random
import os

np.random.seed(42)
random.seed(42)

positive_templates = [
    "Absolutely love this {}! It exceeded all my expectations.",
    "This {} is fantastic. Best purchase I have made all year.",
    "I am so happy with my {} purchase. Works perfectly and looks great.",
    "Great {} for the price. Highly recommend to anyone looking for a quality product.",
    "Five stars! This {} is amazing and arrived quickly.",
    "My {} is wonderful. The quality is outstanding for the price point.",
    "I cannot believe how good this {} is. Totally worth every penny.",
    "Excellent {}! Easy to use, well designed, and durable.",
    "Very satisfied with this {}. Exactly what I needed and more.",
    "This {} is a game changer. So glad I decided to buy it.",
    "Perfect {}! It does everything I wanted and more.",
    "Best {} I have ever owned. The quality is top notch.",
    "I am thrilled with my {}. Shipping was fast and packaging was great.",
    "This {} works like a charm. Easy setup and intuitive controls.",
    "Amazing quality {} for such a reasonable price. Will buy again.",
]

neutral_templates = [
    "The {} is okay. It does the job but nothing special.",
    "Decent {} for the money. Not great but not terrible either.",
    "I have mixed feelings about this {}. It has some good features but also drawbacks.",
    "Average {} performs as expected. No complaints but no surprises.",
    "This {} works fine. It is not the best but it gets the job done.",
    "The {} arrived on time and was as described. Nothing more to say really.",
    "It is an okay {}. I might look for something better in the future.",
    "Not bad but not great either. The {} serves its purpose.",
    "This {} is alright. I have seen better but I have also seen worse.",
    "My {} is functional. It could use some improvements in design though.",
    "The {} does what it is supposed to do. That is about it.",
    "Pretty standard {}. Nothing really stands out about it.",
]

negative_templates = [
    "Terrible {}! It broke within the first week of use.",
    "Very disappointed with this {}. The quality is just not there.",
    "Do not waste your money on this {}. It is poorly made and does not work well.",
    "This {} is a piece of junk. Cheap materials and bad design.",
    "I regret buying this {}. It stopped working after just a month.",
    "Worst {} I have ever purchased. Totally useless product.",
    "The {} is horrible. Customer service was no help either.",
    "I hate this {}. It is nothing like the description said it would be.",
    "Stay away from this {}! It is overpriced and underperforms.",
    "This is the worst {} ever. Save your money and buy something else.",
    "Completely dissatisfied with this {}. Do not recommend it to anyone.",
    "What a waste of money this {} turned out to be. Very poor quality.",
    "Awful {} with terrible build quality. I want my money back.",
    "This {} does not work properly at all. Extremely frustrating experience.",
    "The {} is defective. Tried to get a replacement but no luck.",
]

products = ['wireless headphones', 'smartphone case', 'bluetooth speaker', 'laptop backpack',
            'coffee maker', 'yoga mat', 'running shoes', 'tablet stand',
            'phone charger', 'water bottle', 'desk lamp', 'meal prep containers',
            'portable fan', 'mouse pad', 'screen protector']

misspellings = {'absolutely': 'absolutly', 'amazing': 'amazng', 'fantastic': 'fantaastic',
                'disappointed': 'dissapointed', 'recommend': 'reccomend', 'purchase': 'purchace',
                'quality': 'quallity', 'defective': 'defektive', 'experience': 'experiance',
                'satisfied': 'satisified', 'wonderful': 'wunderful', 'terrible': 'terible',
                'features': 'feautures', 'performance': 'perfomance', 'bought': 'brought'}

n = 2000
sentiments = np.random.choice(['Positive', 'Neutral', 'Negative'], n, p=[0.45, 0.30, 0.25])
reviews = []
sentiments_list = []
review_ids = []

for i in range(n):
    product = random.choice(products)
    sent = sentiments[i]
    if sent == 'Positive':
        template = random.choice(positive_templates)
    elif sent == 'Neutral':
        template = random.choice(neutral_templates)
    else:
        template = random.choice(negative_templates)

    review = template.format(product)

    if random.random() < 0.15:
        wrong_word, correct_word = random.choice(list(misspellings.items()))
        if correct_word in review.lower():
            review = review.lower().replace(correct_word, wrong_word, 1)
            review = review[0].upper() + review[1:] if review else review

    if random.random() < 0.08:
        review = review[:-1] + '!!' if review.endswith('.') else review + '!!'

    reviews.append(review)
    sentiments_list.append(sent)
    review_ids.append(f"R{1000+i:04d}")

df = pd.DataFrame({'review_id': review_ids, 'review_text': reviews, 'sentiment': sentiments_list})

out_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'product_reviews.csv')
df.to_csv(out_path, index=False)
print(f"Generated {out_path} ({len(df)} rows)")
