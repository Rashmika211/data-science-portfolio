import re
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'

    messages = re.split(pattern, data)[1:]

    # Get usernames from messages
    users = []
    msg = []

    for message in messages:
        reply = re.split('([\w\W]+?):\s', message)
        # print(reply)
        if reply[1:]:
            users.append(reply[1])
            msg.append("".join(reply[2:]))
        else:
            users.append('group notification')
            msg.append(reply[0])

    dates = re.findall(pattern, data)


    df = pd.DataFrame({'datetime':dates, 'User':users, 'Message':msg})
    # convert dates type
    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%y, %H:%M - ')

    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month_name()
    df['day'] = df['datetime'].dt.day
    df['day_name'] = df['datetime'].dt.day_name()
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['month_num'] = df['datetime'].dt.month

    # Dropping Non-messages
    df.drop(df[df['Message'] == '<Media omitted>\n'].index, axis = 0, inplace=True)
    df.drop(df[df['Message'] == 'This message was deleted\n'].index, axis = 0, inplace=True)
    df.drop(df[df['User'] == 'group notification'].index, axis = 0, inplace=True)

    # fetch time period
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append(str(hour) + '-' + str('00'))
        else:
            period.append(str(hour) + '-' + str(hour+1))
    df['period'] = period

    # fetch sentiment value
    sentiments=SentimentIntensityAnalyzer()
    df["positive"]=[sentiments.polarity_scores(i)["pos"] for i in df["Message"]]
    df["negative"]=[sentiments.polarity_scores(i)["neg"] for i in df["Message"]]
    df["neutral"]=[sentiments.polarity_scores(i)["neu"] for i in df["Message"]]

    def sentiment(d):
        if d["positive"] >= d["negative"] and d["positive"] >= d["neutral"]:
            return 1
        if d["negative"] >= d["positive"] and d["negative"] >= d["neutral"]:
            return -1
        if d["neutral"] >= d["positive"] and d["neutral"] >= d["negative"]:
            return 0

    df['sentiment'] = df.apply(lambda row: sentiment(row), axis=1)

    return df