import streamlit as st 
import streamlit.components.v1 as components
import requests
import time
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import re
from matplotlib import pyplot as plt
import torch 
import torch.nn as nn
from transformers import AutoTokenizer

class TweetClassifier(nn.Module):
    def __init__(self, bertModel, out_feat, freeze_bert):
        super().__init__()
        D_in, H, D_out = bertModel.config.hidden_size, 50, out_feat
        self.bert = bertModel
        self.classifier = nn.Sequential(
            nn.Linear(D_in,H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )
        if freeze_bert==True:
          for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask )
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits


DB_NAME = 'licenta_political_tweets'
positive_text = "I love Trump because it is a beautiful man"
negative_text = "I hate Trump because it is an ugly man"
fake_text = "Afghanistan IS TAKING CONTROL OF RUSSIA AT football a football match in Bali"
true_text = "President Biden leads America to take control of Russian armies"

def get_all_text(df, col):
    text = ' '.join(str(el) for el in df[col] if el != ' ')
    text = re.sub(' +', ' ', text)
    return text


@st.cache_resource(show_spinner=True)
def show_wordcloud(df, col):
    wc = WordCloud(collocations=False, stopwords=STOPWORDS,
               background_color="white", max_words=1000,
               max_font_size=256, random_state=42,
               width=1400, height=400)

    text=get_all_text(df, col)
    wc.generate(text)
    return wc
    

def load_misinformation_model():
    model_misinformation = torch.load('model_misinf_v1.pt',map_location='cpu')
    tokenizer = AutoTokenizer.from_pretrained('m-newhauser/distilbert-political-tweets')
    model_misinformation.eval()
    return model_misinformation, tokenizer

def load_sentiment_model():
    sentiment_model = torch.load('model_sentiment140_v2_final.pt',map_location='cpu')
    tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
    
    sentiment_model.eval()
    return sentiment_model, tokenizer

def preprocess(text):
    new_text = []
    if type(text)==float:
      print(text)
    for t in text.split(" "):
        t = '@USER' if t.startswith('@') and len(t) > 1 else t
        t = 'HTTPURL' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def id2label_sentiment(id):
    if id==1:
        return 'Positive'
    elif id== 0:
        return 'Negative'
    else:
        return 'Neutral'


def id2label_misinformation(id):
    if id==0:
        return 'Fake'
    elif id== 1:
        return 'True'
    else:
        return 'not known'


def tokenize_function(text, tokenizer):
    tok = tokenizer(text,add_special_tokens=True, padding="max_length", max_length = 128,truncation=True, return_tensors="pt")
    return tok['input_ids'], tok['attention_mask']


def infer_from_model(text, model, tokenizer, model_type):
    cleaned_text = preprocess(text)
    inputs_ids, attention_mask = tokenize_function(cleaned_text, tokenizer=tokenizer)
    logits = model(inputs_ids, attention_mask)
    print(logits)
    prediction = torch.argmax(torch.abs(logits), dim=1)
    if model_type == 'sentiment':
        return id2label_sentiment(prediction.item())
    else:
        return id2label_misinformation(prediction.item())


def color_df_misinf(val):
    color = 'green' if val=='True' else 'red'
    return f'background-color: {color}'

def color_df_sentiment(val):
    color = 'pink' if val=='Positive' else 'blue'
    return f'color: {color}'

def color_df_all(val):
    print('VAL:')
    print(val)
    color_misinf =  'green' if val['truthness']=='True' else 'red'
    color_polarity = 'yellow' if val['polarity']=='Positive' else 'bleu'
    return f'background-color: {color_misinf}, color: {color_polarity}'

df_russia = pd.read_csv('EXTRA_RussianPropagandaSubset.csv')
df_russia = df_russia.drop(columns=['ID'])
df_russia = df_russia.drop_duplicates(subset=['Text'])

st.set_page_config(page_icon="🐤", page_title="Political Tweets Analyzer")
st.sidebar.image("russia_propaganda_on_twitter.jpg", use_column_width=True)

model_misinformation, tokenizer_misinformation = load_misinformation_model()
model_sentiment, tokenizer_sentiment = load_sentiment_model()
    
st.sidebar.subheader('Analyize your text')
text_introduced = st.sidebar.text_input('Type', placeholder=positive_text)
check_polarity = st.sidebar.button(label='Check Polarity')
check_truth = st.sidebar.button(label='Check Misinformation')

if check_polarity:
    result =  infer_from_model(text_introduced, model_sentiment, tokenizer_sentiment,'sentiment')
    st.sidebar.write(result)
if check_truth:
    result = infer_from_model(text_introduced, model_misinformation, tokenizer_misinformation,'misinformation')
    st.sidebar.write(result)
    

st.write('<base target="_blank">', unsafe_allow_html=True)
prev_time = [time.time()]

a, b = st.columns([1, 10])

with a:
    st.text("")
    st.image("logoOfficial.png", width=50)
with b:
    st.title("Political Tweets Analyzer")

st.header('Russia Propaganda')
st.dataframe(df_russia)

treshold = int(st.number_input('Insert the maximum number of tweets to analyze',min_value=1,max_value=500,step=5))
if treshold > 0:
    df_russia_limited = df_russia[:treshold]
    cols_shows = st.columns([1,1,1])
    show_polarity_for_batch = cols_shows[0].button(label="Compute polarity")
    show_misinf_for_batch = cols_shows[1].button(label="Compute truthness")
    show_analysis_for_batch = cols_shows[2].button(label='Compute both')
    
    if show_polarity_for_batch:
        df_russia_limited['polarity'] = df_russia_limited.apply(lambda x: infer_from_model(x['Text'],model_sentiment,tokenizer_sentiment, 'sentiment'),axis=1)
        st.dataframe(df_russia_limited.style.applymap(color_df_sentiment, subset=['polarity']))
    
    if show_misinf_for_batch:
        df_russia_limited['truthness'] = df_russia_limited.apply(lambda x: infer_from_model(x['Text'],model_misinformation,tokenizer_misinformation, 'misinformation'),axis=1)
        st.dataframe(df_russia_limited.style.applymap(color_df_misinf, subset=['truthness']))
    
    if show_analysis_for_batch:
        df_russia_limited['truthness'] = df_russia_limited.apply(lambda x: infer_from_model(x['Text'],model_misinformation,tokenizer_misinformation, 'misinformation'),axis=1)
        df_russia_limited['polarity'] = df_russia_limited.apply(lambda x: infer_from_model(x['Text'],model_sentiment,tokenizer_sentiment, 'sentiment'),axis=1)
        df_russia_limited = df_russia_limited.style.applymap(color_df_misinf, subset=['truthness'])
        df_russia_limited = df_russia_limited.applymap(color_df_sentiment, subset=['polarity'])
        
        st.dataframe(df_russia_limited)

show_wc = st.checkbox('Show Wordcloud from all Tweets about Russia Propaganda')
if show_wc:
    wc = show_wordcloud(df_russia, 'Text')
    fig = plt.figure(figsize=(20, 10), facecolor='k')
    plt.title(f'Wordcloud')
    plt.imshow(wc, interpolation='bilInear')
    plt.axis("off")
    plt.tight_layout(pad=0)   
    st.pyplot(fig)
