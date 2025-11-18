"""
Streamlit app: Fake Account Detector - Improved styling and confidence display.
Preserves all backend logic and model usage from your original file.
Changes:
 - Fix overlapping colors and layout spacing.
 - Replace multiple/confusing progress bars with a single styled confidence bar + numeric display.
 - Cleaner metric presentation and consistent card backgrounds.
 - Minor defensive checks (model clients) retained.
"""

import os
import datetime
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import streamlit as st
from transformers import BertTokenizer, BertModel
import xgboost as xgb
import praw
import tweepy

# ---------- Minimal CSS tuned to avoid overlaps ----------
def local_css(css: str):
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

CUSTOM_CSS = """
/* Base */
body, .stApp { background: #0f1214 !important; color: #ECEFF1 !important; }
.block-container { padding-top: 20px !important; padding-left: 28px !important; padding-right: 28px !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #15181A !important; padding: 24px 18px !important; }n
/* Headings */
h1 { color:#ECEFF1; font-weight:700; }
h2 { color:#ECEFF1; }

/* Card-like containers */
.stCard, .stExpander, .stMetric, .stAlert {
  background: #16181A !important;
  border-radius: 10px !important;
  padding: 12px 16px !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.35) !important;
}

/* Inputs */
textarea, input[type="text"], input[type="number"] {
  background:#1f2326 !important; color:#ECEFF1 !important; border-radius:8px !important;
  padding:10px !important; border:1px solid #2b2f32 !important;
}

/* Buttons */
.stButton button {
  background:#00B894 !important; color:#071216 !important; height:46px; border-radius:10px;
  font-weight:700; border:none; padding: 0 18px;
}
.stButton button:hover { filter:brightness(1.03); transform: translateY(-1px); }

/* Metrics row */
.metric-label { color:#9AA0A6; font-size:0.9rem; margin-bottom:6px; }
.metric-value { color:#ECEFF1; font-size:1.6rem; font-weight:800; }

/* Info banner */
.info-banner {
  background:#2f445a !important; color:#e6f0fb !important; padding:10px 14px; border-radius:8px;
  margin: 12px 0;
}

/* Confidence bar (custom) */
.confidence-wrap { background:#111418; padding:12px; border-radius:10px; }
.confidence-label { color:#3B82F6; font-weight:700; margin-bottom:8px; }
.confidence-bar-bg {
  background:#22272a; border-radius:8px; height:18px; position:relative; overflow:hidden;
}
.confidence-bar-fill {
  background: linear-gradient(90deg,#00B894,#00D9AD); height:100%; width:0%;
  border-radius:8px; transition: width 0.7s ease;
}
.confidence-value { float:right; font-weight:700; color:#ECEFF1; margin-left:12px; }

/* Small screens / responsiveness */
@media (max-width: 900px) {
  .block-container { padding-left:12px !important; padding-right:12px !important; }
}
"""

local_css(CUSTOM_CSS)

# ---------- Page ----------
st.set_page_config(page_title="Fake Account Detector - Inference GUI", layout="wide", initial_sidebar_state="expanded")
st.title("Fake Account Detector - Inference GUI")
st.write("Welcome. Select a mode and provide input to analyze an account or paste text.")

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------- Paths & device ----------
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'models')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- API credentials (placeholders preserved) ----------
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID") or 'JRNjjalIhYOQ3JR7AjFmkA'
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET") or 'k2IyGQ9OOA-6vNcp58jGSmn3KpQ4Uw'
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT") or 'fake_account_detector/1.0 by Winter_Lingonberry60'
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN") or "AAAAAAAAAAAAAAAAAAAAAOli4AEAAAAAM38OxA6vo2D8m2p7poBCA6kTnuY%3D4vEKut29Do1H1jCyaddqZdvrcTkzAM0fWpnGAKPR4t4keSeksS"

# ---------- Model architecture (unchanged) ----------
class CNN(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(hidden_dim * (input_dim // 2), 64)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ---------- BERT helper (robust to empty) ----------
def get_bert_embeddings(texts, tokenizer, bert_model, device, batch_size=16):
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, (pd.Series, np.ndarray)):
        texts = list(texts)
    else:
        texts = list(texts)

    texts = [t if (t and str(t).strip()) else "[EMPTY]" for t in texts]

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_emb)
    if embeddings:
        return np.vstack(embeddings)
    return np.zeros((len(texts), 768), dtype=np.float32)

# ---------- Cached loaders (preserve) ----------
@st.cache_resource(show_spinner=False)
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()
    return tokenizer, model

@st.cache_resource(show_spinner=False)
def load_cnn_model():
    m = CNN().to(device)
    pth = os.path.join(OUTPUT_PATH, 'cnn_model.pth')
    if os.path.exists(pth):
        m.load_state_dict(torch.load(pth, map_location=device))
        m.eval()
    else:
        st.warning(f"CNN model file not found at {pth}. Predictions will not work until model is available.")
    return m

@st.cache_resource(show_spinner=False)
def load_xgb_model():
    model = xgb.XGBClassifier()
    path = os.path.join(OUTPUT_PATH, 'xgb_model_tuned.json')
    if os.path.exists(path):
        model.load_model(path)
    else:
        st.warning(f"XGBoost model file not found at {path}. Predictions will not work until model is available.")
    return model

@st.cache_resource(show_spinner=False)
def load_reddit_client(client_id, client_secret, user_agent):
    try:
        reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
        return reddit
    except Exception as e:
        st.error(f"Error initializing Reddit client: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_twitter_client(bearer_token):
    try:
        client = tweepy.Client(bearer_token)
        return client
    except Exception as e:
        st.error(f"Error initializing Twitter client: {e}")
        return None

tokenizer, bert_model = load_bert_model()
cnn_model = load_cnn_model()
xgb_model = load_xgb_model()
reddit = load_reddit_client(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT)
twitter_client = load_twitter_client(TWITTER_BEARER_TOKEN)

# ---------- Feature preprocessing (unchanged logic) ----------
def preprocess_input_features(bio_text: str, metadata_dict: dict, platform: str):
    bert_embedding = get_bert_embeddings(bio_text, tokenizer, bert_model, device)
    with torch.no_grad():
        cnn_features = cnn_model(torch.tensor(bert_embedding, dtype=torch.float32).to(device)).cpu().numpy()

    num_cols = [
        'followers_count', 'following_count', 'post_count', 'username_length',
        'username_digit_count', 'mean_likes', 'mean_comments', 'mean_hashtags',
        'upload_interval_std'
    ]
    processed = []
    for col in num_cols:
        value = metadata_dict.get(col, 0)
        try:
            processed.append(np.log1p(max(0.0, float(value))))
        except Exception:
            processed.append(0.0)
    processed_array = np.array(processed).reshape(1, -1)

    platform_cols = ['instagram', 'twitter']
    platform_one_hot = np.zeros(len(platform_cols))
    if platform in platform_cols:
        platform_one_hot[platform_cols.index(platform)] = 1
    platform_one_hot_array = platform_one_hot.reshape(1, -1)

    final = np.hstack([cnn_features, processed_array, platform_one_hot_array])
    return final

def safe_predict_proba(model, features):
    try:
        return model.predict_proba(features)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return np.array([1.0, 0.0])

# ---------- UI layout ----------
st.sidebar.title("Mode")
mode = st.sidebar.radio("", ['Manual text input', 'Reddit username', 'Twitter username'])

if mode == 'Manual text input':
    st.sidebar.subheader("Manual Text Input")
    text_input = st.sidebar.text_area("Enter text (bio or post)", height=180)

    st.sidebar.markdown("--- Optionally provide metadata---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.number_input("Followers Count", min_value=0, value=0, key='manual_followers')
        st.number_input("Following Count", min_value=0, value=0, key='manual_following')
        st.number_input("Post Count", min_value=0, value=0, key='manual_posts')
    with col2:
        st.number_input("Username Length", min_value=0, value=0, key='manual_uname_len')
        st.number_input("Username Digit Count", min_value=0, value=0, key='manual_uname_digits')
        st.number_input("Mean Likes (if applicable)", min_value=0.0, value=0.0, key='manual_mean_likes')

    predict_btn = st.sidebar.button("Predict")
elif mode == 'Reddit username':
    st.sidebar.subheader("Reddit Username")
    reddit_username = st.sidebar.text_input("Enter Reddit Username (no u/)")
    num_reddit_submissions = st.sidebar.number_input("Number of Submissions (Posts) to fetch", min_value=1, max_value=100, value=10, key='num_reddit_submissions')
    num_reddit_comments = st.sidebar.number_input("Number of Comments to fetch", min_value=1, max_value=100, value=10, key='num_reddit_comments')
    predict_btn = st.sidebar.button("Analyze Reddit User")
else: # Twitter username
    st.sidebar.subheader("Twitter Username")
    twitter_username = st.sidebar.text_input("Enter Twitter Username (no @)")
    num_twitter_tweets = st.sidebar.number_input("Number of Tweets to fetch", min_value=1, max_value=100, value=20, key='num_twitter_tweets')
    predict_btn = st.sidebar.button("Analyze Twitter User")

main_col, right_col = st.columns([2.5, 1], gap="large")

with main_col:
    status_box = st.container()
    confidence_box = st.container()
    info_box = st.container()
    st.markdown("### Activity sample / Timeline")
    timeline_box = st.empty()  # placeholder

with right_col:
    st.markdown("### Key Metrics")
    k1, k2, k3 = st.columns([1,1,1])
    followers_disp = k1.metric("Followers", "—")
    following_disp = k2.metric("Following", "—")
    tweets_disp = k3.metric("Tweets", "—")

# ---------- Helper to render single confidence bar + numeric ----------
def render_confidence(label: str, value: float, container):
    """
    Renders a single confidence bar with numeric value (value between 0 and 1).
    """
    pct = max(0.0, min(1.0, float(value)))
    pct_display = f"{pct*100:.1f}%"
    # Write label and numeric
    container.markdown(f'<div class="confidence-wrap"><div class="confidence-label">{label} <span style="float:right" class="confidence-value">{pct_display}</span></div>'
                       f'<div class="confidence-bar-bg"><div class="confidence-bar-fill" style="width: {pct*100}%;"></div></div></div>',
                       unsafe_allow_html=True)

# ---------- Prediction logic (preserve behavior, improved displays) ----------
if predict_btn:
    # reset displays
    status_box.empty(); confidence_box.empty(); info_box.empty(); timeline_box.empty()

    if mode == 'Manual text input':
        if not (text_input and text_input.strip()):
            st.sidebar.warning("Please paste some text to analyze.")
        else:
            with st.spinner("Processing and predicting..."):
                # manual_metadata intentionally contains zeros as requested by the user
                # The UI inputs are for authenticity but not backend prediction in this mode.
                features = preprocess_input_features(text_input, {
                    'followers_count': 0, 'following_count': 0, 'post_count': 0,
                    'username_length': 0, 'username_digit_count': 0,
                    'mean_likes': 0.0, 'mean_comments': 0.0, 'mean_hashtags': 0.0,
                    'upload_interval_std': 0.0
                }, 'twitter')  # keep same default
                proba = safe_predict_proba(xgb_model, features)
                fake_proba = float(proba[1])

            label = 'Fake/Automated' if fake_proba > 0.5 else 'Legitimate/Human'
            status_box.markdown(f"## **{label}**")
            render_confidence("Confidence (Fake)", fake_proba, confidence_box)
            info_box.info(f"Probability of being Fake/Automated: {fake_proba:.4f}")

            st.success("Prediction Complete!")

            # Display the manually entered metadata in the right column for user feedback
            followers_disp.metric("Followers", str(st.session_state['manual_followers']))
            following_disp.metric("Following", str(st.session_state['manual_following']))
            tweets_disp.metric("Posts", str(st.session_state['manual_posts']))

    elif mode == 'Reddit username':
        if not reddit_username or not reddit_username.strip():
            st.sidebar.warning("Please provide a Reddit username.")
        else:
            if reddit is None:
                st.sidebar.warning("Reddit API not initialized. Check credentials.")
            else:
                with st.spinner(f"Fetching u/{reddit_username} and predicting..."):
                    try:
                        redditor = reddit.redditor(reddit_username)
                        redditor._fetch_data()
                        comment_karma = getattr(redditor, "comment_karma", 0)
                        link_karma = getattr(redditor, "link_karma", 0)
                        created_utc = getattr(redditor, "created_utc", None)
                        account_age_days = (datetime.datetime.now() - datetime.datetime.fromtimestamp(created_utc)).days if created_utc else 0

                        bio_texts = []
                        # Use separate limits for submissions and comments
                        num_submissions_to_fetch = st.session_state['num_reddit_submissions']
                        num_comments_to_fetch = st.session_state['num_reddit_comments']

                        for submission in redditor.submissions.new(limit=num_submissions_to_fetch):
                            if getattr(submission, "is_self", False):
                                bio_texts.append((getattr(submission, "title", "") or "") + " " + (getattr(submission, "selftext", "") or ""))
                            else:
                                bio_texts.append(getattr(submission, "title", "") or "")
                        for comment in redditor.comments.new(limit=num_comments_to_fetch):
                            bio_texts.append(getattr(comment, "body", "") or "")
                        bio_text = " ".join(bio_texts).strip()

                        st.info(f"Length of extracted bio_text before BERT: {len(bio_text)} characters")

                        reddit_metadata = {
                            'followers_count': comment_karma,
                            'following_count': 0,
                            'post_count': link_karma,
                            'username_length': len(reddit_username),
                            'username_digit_count': sum(c.isdigit() for c in reddit_username),
                            'mean_likes': 0, 'mean_comments': 0, 'mean_hashtags': 0, 'upload_interval_std': 0
                        }

                        features = preprocess_input_features(bio_text, reddit_metadata, 'reddit')
                        proba = safe_predict_proba(xgb_model, features)
                        fake_proba = float(proba[1])

                        label = 'Fake/Automated' if fake_proba > 0.5 else 'Legitimate/Human'
                        status_box.markdown(f"## **{label}**")
                        render_confidence("Confidence (Fake)", fake_proba, confidence_box)
                        info_box.info(f"Probability of being Fake/Automated: {fake_proba:.4f}")

                        followers_disp.metric("Comment Karma (proxy)", str(comment_karma))
                        following_disp.metric("Following", "—")
                        tweets_disp.metric("Link Karma (proxy)", str(link_karma))

                        st.write("-- Reddit User Data Found --")
                        st.write(f"Comment Karma: {comment_karma}")
                        st.write(f"Link Karma: {link_karma}")
                        st.write(f"Account Age (days): {account_age_days}")
                        st.write(f"Extracted Bio Text (first 200 chars): {bio_text[:200]}...")
                        st.success("Prediction Complete!")

                    except Exception as e:
                        st.error(f"Could not fetch Reddit user data for '{reddit_username}'. Error: {e}")
                        st.info("Ensure the username is correct and Reddit API credentials are valid.")

    else:  # Twitter username
        if not twitter_username or not twitter_username.strip():
            st.sidebar.warning("Please provide a Twitter username.")
        else:
            if twitter_client is None:
                st.sidebar.warning("Twitter API not initialized. Check Bearer Token.")
            else:
                with st.spinner(f"Fetching @{twitter_username} and predicting..."):
                    try:
                        user_response = twitter_client.get_user(username=twitter_username,
                                                           user_fields=['description', 'public_metrics', 'created_at'])
                        user = getattr(user_response, "data", None)
                        if not user:
                            st.error(f"Twitter user '{twitter_username}' not found or inaccessible.")
                        else:
                            followers_count = user.public_metrics.get('followers_count', 0)
                            following_count = user.public_metrics.get('following_count', 0)
                            tweet_count = user.public_metrics.get('tweet_count', 0)

                            # Get recent tweets to form bio_text
                            tweets_response = twitter_client.get_users_tweets(id=user.id,
                                                                              tweet_fields=['text'],
                                                                              max_results=st.session_state['num_twitter_tweets'])
                            recent_tweets_text = []
                            if tweets_response.data:
                                for tweet in tweets_response.data:
                                    recent_tweets_text.append(tweet.text)

                            bio_text = user.description or ""
                            if recent_tweets_text:
                                bio_text += " " + " ".join(recent_tweets_text)
                            bio_text = bio_text.strip()

                            username_length = len(user.username)
                            username_digit_count = sum(c.isdigit() for c in user.username)
                            created_at = getattr(user, "created_at", None)
                            account_age_days = (datetime.datetime.now(datetime.timezone.utc) - created_at).days if created_at else 0

                            st.info(f"Length of extracted bio_text before BERT: {len(bio_text)} characters")

                            twitter_metadata = {
                                'followers_count': followers_count,
                                'following_count': following_count,
                                'post_count': tweet_count,
                                'username_length': username_length,
                                'username_digit_count': username_digit_count,
                                'mean_likes': 0, 'mean_comments': 0, 'mean_hashtags': 0, 'upload_interval_std': 0
                            }

                            features = preprocess_input_features(bio_text, twitter_metadata, 'twitter')
                            proba = safe_predict_proba(xgb_model, features)
                            fake_proba = float(proba[1])

                            label = 'Fake/Automated' if fake_proba > 0.5 else 'Legitimate/Human'
                            status_box.markdown(f"## **{label}**")
                            render_confidence("Confidence (Fake)", fake_proba, confidence_box)
                            info_box.info(f"Probability of being Fake/Automated: {fake_proba:.4f}")

                            followers_disp.metric("Followers", str(followers_count))
                            following_disp.metric("Following", str(following_count))
                            tweets_disp.metric("Tweets", str(tweet_count))

                            st.write("-- Twitter User Data Found --")
                            st.write(f"Followers Count: {followers_count}")
                            st.write(f"Following Count: {following_count}")
                            st.write(f"Tweet Count: {tweet_count}")
                            st.write(f"Account Age (days): {account_age_days}")
                            st.write(f"Bio Text (first 200 chars): {bio_text[:200]}...")
                            st.success("Prediction Complete!")

                    except tweepy.errors.TweepyException as e:
                        st.error(f"Twitter API error: {e}")
                        st.info("Ensure username is correct and Bearer Token has access.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
