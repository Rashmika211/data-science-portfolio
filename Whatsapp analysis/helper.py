import emoji
import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud
from stop_words import get_stop_words
stop_words = get_stop_words('en')
from collections import Counter



def fetch_stats(selected_user, df):
    # fetch no. of messages
    num_messages =  df.shape[0]
    # fetch word count
    all_words = []
    for message in df['Message']:
        all_words.extend(message.split())
    # fetch emoji count
    emoji_list = []
    for message in df['Message']:
        for char in message:
            if char in emoji.EMOJI_DATA:
                emoji_list.append(char)
    # fetch Links count
    extractor = URLExtract()
    links = []
    for message in df['Message']:
        links.extend(extractor.find_urls(message))


    return num_messages, len(all_words), len(emoji_list), len(links)


def most_active_usrs(df):
    most_active = df['User'].value_counts().head(5)
    per_active = round((df['User'].value_counts() / df.shape[0]) * 100, 2).rename('percentage active')
    return most_active, per_active



def get_wordcloud(selected_user, df):
    wc = WordCloud(stopwords = stop_words, width=500, height=500, min_font_size=10, background_color='black')
    wc_gen = wc.generate(df['Message'].str.cat(sep=' '))
    return wc_gen



def most_common_words(selected_user, df):
    words = []
    for message in df['Message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    return pd.DataFrame(Counter(words).most_common(20))


def get_emoji(selected_user, df):
    emoji_list = []
    for message in df['Message']:
        for char in message:
            if char in emoji.EMOJI_DATA:
                emoji_list.append(char)

    emoji_df = pd.DataFrame(Counter(emoji_list).most_common(len(Counter(emoji_list)))).rename(columns = {0:'emoji', 1:'count'})
    emoji_df['percentage'] = round(emoji_df['count'] / emoji_df['count'].sum() * 100, 2)

    return emoji_df



def monthly_timeline(selected_user, df):
    timeline = df.groupby(['year', 'month_num', 'month']).count()['Message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i]+'-'+str(timeline['year'][i]))

    timeline['time'] = time
    return timeline


def weekly_activity_map(selected_df, df):
    return df['day_name'].value_counts()

def month_activity_map(selected_df, df):
    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    pt = df.pivot_table(index='day_name', columns = 'period', values = 'Message', aggfunc='count').fillna(0)

    return pt

# -1 => Negative
# 0 => Neutral
# 1 => Positive
def sentiment_wordcloud(selected_user,df,k):
    temp = df.copy()
    # Dimensions of wordcloud
    wc = WordCloud(stopwords=stop_words,width=500,height=500,min_font_size=10,background_color='white')

    temp['Message'] = temp['Message'][temp['sentiment'] == k]

    # Word cloud generated
    df_wc = wc.generate(temp['Message'].str.cat(sep=" "))
    return df_wc

def percentage(df,k):
    df = round((df['User'][df['sentiment']==k].value_counts().head(5) / df[df['sentiment']==k].shape[0]).head(5) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return df


















