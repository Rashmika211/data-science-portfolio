import streamlit as st
import preprocess
import re
import stats
import matplotlib.pyplot as plt
import numpy as np
import helper
import seaborn as sns


# Title of Sidebar
st.sidebar.title('WhatsApp Chat Analyzer')
# Uploading file
upload_file = st.sidebar.file_uploader("Choose a WhatsApp chat file")

if upload_file is not None:
    # extracting uploaded data
    bytes_data = upload_file.getvalue()
    # saving data
    data = bytes_data.decode("utf-8")
    # preprocessing the data
    df = preprocess.preprocess(data)
    # fetch all users
    user_list = df['User'].unique().tolist()
    # user_list.remove('group notification')
    # user_list.sort()
    user_list.insert(0, 'Overall')
    st.sidebar.text('''

    ''')
    selected_user = st.sidebar.selectbox("Select a user below to Analyze", user_list)

    if st.sidebar.button('Analyze'):

        if selected_user!='Overall':
            df = df[df['User'] == selected_user]

        # st.dataframe(df)
        st.title('Top Statistics of Chat')
    # fetch stats
        col1, col2, col3, col4 = st.columns(4)
        num_messages, word_count, emoji_count, links_count = helper.fetch_stats(selected_user, df)
        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(word_count)
        with col3:
            st.header("Total Emojis")
            st.title(emoji_count)
        with col4:
            st.header("Total Links")
            st.title(links_count)


    # Timeline
        timeline = helper.monthly_timeline(selected_user, df)
        st.title('Monthly Timeline')

        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['Message'])
        plt.xticks(rotation = 'vertical')
        st.pyplot(fig)


    # Activity map
        st.title('Activity Map')

        col1, col2 = st.columns(2)
        with col1:
            st.title('Most busy day')
            week_activity = helper.weekly_activity_map(selected_user, df)

            fig, ax = plt.subplots()
            ax.bar(week_activity.index, week_activity.values)
            plt.xticks(rotation = 'vertical')
            st.pyplot(fig)
        with col2:
            st.title('Most busy month')
            monthly_activity = helper.month_activity_map(selected_user, df)

            fig, ax = plt.subplots()
            ax.bar(monthly_activity.index, monthly_activity.values, color = 'pink')
            plt.xticks(rotation = 'vertical')
            st.pyplot(fig)


    # Activity heatmap
        pt = helper.activity_heatmap(selected_user, df)
        st.title('Activity Heatmap')
        fig, ax = plt.subplots()
        ax = sns.heatmap(pt)
        plt.xlabel('''
        Time Period(24 hours)''')
        plt.ylabel('Day')
        st.pyplot(fig)



    # Most active users in group

        if selected_user == 'Overall':
            st.title('Most Active Users')
            active, per = helper.most_active_usrs(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)
            with col1:
                ax.bar( active.index, active.values, color = ['red','green','blue','orchid','pink'])
                plt.xticks(rotation = 'vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(per)



    # WordCloud
        st.title('Wordcloud')

        word_cloud = helper.get_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(word_cloud)
        plt.axis('off')
        st.pyplot(fig)


    # Most common words
        col1, col2 = st.columns(2)
        with col1:
            most_common_df = helper.most_common_words(selected_user, df)
            st.title('Most Common words')
            fig, ax = plt.subplots()
            ax.barh(most_common_df[0], most_common_df[1])
            st.pyplot(fig)

    # emoji analysis
        with col2:
            emoji_df = helper.get_emoji(selected_user, df)
            st.title('Emoji Analysis')
            st.dataframe(emoji_df)


    # Most Positive,Negative,Neutral User...
        if selected_user == 'Overall':
            st.title('Most Contributed Users ')
            col1,col2,col3 = st.columns(3)

            with col1:
                st.title('Positive')
                x = helper.percentage(df, 1)

                # Displaying
                st.dataframe(x)
            with col2:
                st.title('Neutral')
                y = helper.percentage(df, 0)

                # Displaying
                st.dataframe(y)
            with col3:
                st.title('Negative')
                z = helper.percentage(df, -1)

                # Displaying
                st.dataframe(z)



    # Sentiment wordcloud
        st.title('Sentiment Wordcloud')
        col1,col2,col3 = st.columns(3)

        with col1:
            st.title('Positive')
            df_wc = helper.sentiment_wordcloud(selected_user, df, 1)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            ax.axis('off')
            st.pyplot(fig)
        with col2:
            st.title('Neutral')
            df_wc = helper.sentiment_wordcloud(selected_user, df, 0)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            ax.axis('off')
            st.pyplot(fig)
        with col3:
            st.title('Negative')
            df_wc = helper.sentiment_wordcloud(selected_user, df, -1)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            ax.axis('off')
            st.pyplot(fig)














