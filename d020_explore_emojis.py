# | # Emoji Sentiment Analysis
# |
# | In his notebook I explore how emojis are related to the sentiment in each comment.
# |
# | Just around 5% of the comments contain emojis.
# | The most common emojis in the training set are: 🤣 🤔 🤩 🤘 🤗 🥰.
# |
# | Messages may have the same emoji repeated several times but,
# | in general, they have just one emoji.
# | Some comments have 2 emojis and very few have 3.
# |
# | Individual emoji counts are very small within each of the sentiment groups
# | and trends cannot be guaranteed. Said that,
# | 🤩 seems to be associated with Positive messages while
# | 🤣 🤔 are more common in Negative or Mixed feelings.
# | 🤘 seems to be a good marker of non Tamil languages.
# | Nevertheless, none of the emojis on its own seems to be able to discriminate among groups.
# | Moreover, as very few comments contain 2 or more emojis we will not be able
# | to fit complex models to predict sentiment just based on emojis.
# |
# | In __conclusion__, emojis will incorporate information to any predictive model we may want to use for our sentiment analysis.
# | Hence, it will be a good practice then to try to incorporate them in our NLP modeling.
# | But, on their own, they will not yield a good model, first because they are only in 5% of the messages
# | so we will not be able to predict the 95% of messages that do not have emojis.
# | Second because they do not have enough predictive power on their own
# | and they are generally not combined in the messages so we cannot create a more complex signal.

import emojis
import datasets
import pandas as pd
import plotly.express as px

datasets.logging.set_verbosity_error()

# | Load data
dataset = datasets.load_dataset('tamilmixsentiment')
dataset

# | class labels
class_labels = dataset['train'].features['label'].names
class_labels

# | convert to pandas dataframe
data = dataset['train'].to_pandas()
data

# | There are many non ascii characters in the text
data['non_ascii'] = 127 < data['text'].apply(lambda x: max([ord(character) for character in list(x)]))
data['non_ascii'].value_counts()
data['non_ascii'].value_counts(normalize=True)
data[data.non_ascii]

# | In particular many emojis
data['emojis'] = data['text'].apply(emojis.get).apply(list)  # emojis.get returns a set
data['n_emojis'] = data['emojis'].apply(len)
data['has_emojis'] = 0 < data['n_emojis']

# | Almost 5% of the comments contain an emoji,
data['has_emojis'].value_counts()

# |
data['has_emojis'].value_counts(normalize=True)

# | but most of them contain just one icon.
# | Few comment may contain 2 different emojis and very little of them contain 3 emojis.
pd.DataFrame({
    'counts': data['n_emojis'].value_counts(),
    'proportions': data['n_emojis'].value_counts(normalize=True),
}).rename_axis('N of emojis')

# | let's see some of the messages with emojis
data[data.has_emojis]


# | Let's count the most common emojis
emoji_count = pd.Series(
    data['emojis'].sum()
).value_counts()
emoji_count


# | And let's count within each of our class labels.
# | We first create dummy variables for each emoji
for e in emoji_count.index:
    data[e] = data['emojis'].apply(lambda x: e in x).astype(int)

pd.testing.assert_series_equal(data[emoji_count.index].sum(), emoji_count)  # just a check

data

# | Now we are ready to group by class label.
# | The table below shows the counts of emojis within each sentiment class
emoji_by_label = data.groupby('label')[emoji_count.index].sum().T
emoji_by_label.columns = class_labels
emoji_by_label


# | and explore proportions
# 100 * emoji_by_label.div(emoji_by_label.sum(axis=1), axis=0)  # not interesting as the classes are imbalanced
emoji_proportions = 100 * emoji_by_label / emoji_by_label.sum()
emoji_proportions


# | lets show it in a bar plot
df = emoji_proportions.unstack().reset_index()
df.columns = ['label', 'emoji', 'proportion']

fig = px.bar(df.iloc[::-1], y="emoji", x="proportion", color="label", barmode="group", orientation='h')
fig.update_layout(autosize=False, width=1000, height=2000)
fig.show()
fig.show('png')
