#| # Compare data sources

#| The __Tamilmixsentiment datset__ presented in
#| [Chakravarthi et. al](https://aclanthology.org/2020.sltu-1.28.pdf)
#| was firstly contributed to the
#| [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/TamilSentiMix)
#| and later published in the [Hugging Face Data Repository](https://huggingface.co/datasets/tamilmixsentiment).

#|In this notebook I compare both datsets and make sure there are no differences between them.

import datasets
import pandas as pd

datasets.logging.set_verbosity_error()

#| Read from Hugging Face
ds = datasets.load_dataset('tamilmixsentiment')
ds

#| and merge into a pandas dataframe
hf = pd.concat([v.to_pandas() for k, v in ds.items()])
hf

#| recode labels
hf['label'] = hf['label'].replace(dict(enumerate(ds['test'].features['label'].names)))
print(hf['label'].value_counts())

#| clean whites
hf['text'] = hf['text'].str.strip()

#| Read from UCI
df = pd.read_csv(
    'http://archive.ics.uci.edu/ml/machine-learning-databases/00610/Tamil_first_ready_for_sentiment.csv',
    sep='\t', names=['label', 'text']
)

#| some cleaning
df['label'] = df['label'].str.strip()
df['text'] = df['text'].str.strip()

#| and reorder columns
df = df[hf.columns]
df

#| and recode some labels
df['label'] = df['label']  # .replace('unknown_state', 'Neutral').replace('not-Tamil', 'Other_language')
print(df['label'].value_counts())


#| Lets arrange the datasets in the same order
hf = hf.sort_values(hf.columns.tolist()).reset_index(drop=True)
df = df.sort_values(df.columns.tolist()).reset_index(drop=True)

#| and compare both dataframes.
(hf == df).all()

#|
hf

#|
df

#| In conclusion, both datasets are the same after recoding the labels and stripping white spaces
