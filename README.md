# NLP sentiment analysis of Tamil and English mixed text 

In his notebook I create a fine-tuned model for sentiment analysis of text written in mixed a mixture of Tamil and English language.

I use a __weighted__ version of the loss function to account for the great imbalance in the classes of the dataset.
I also tune the tokenizer to recognize __emojis__ as those are generally relevant to express sentiment.
My model is based on a pre-trained multilingual model which includes Tamil and English; 
this should help in the text classification process.

The dataset was firs published in the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/TamilSentiMix)
and later published in the [Hugging Face Data Repository](https://huggingface.co/datasets/tamilmixsentiment)
from where I read it in this notebook.
The original paper explaining the data set can be found at:
<https://aclanthology.org/2020.sltu-1.28.pdf>



## Links

- Dataset at UCI Machine Learning Repository <http://archive.ics.uci.edu/ml/datasets/TamilSentiMix>
- Dataset at huggingface <https://huggingface.co/datasets/tamilmixsentiment>
- PDF paper: <https://aclanthology.org/2020.sltu-1.28.pdf>
