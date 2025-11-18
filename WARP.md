# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a **Fake Account Detector** machine learning application built with Streamlit. It uses a hybrid model architecture combining BERT embeddings, CNN feature extraction, and XGBoost classification to detect fake/automated accounts across social media platforms (Reddit, Twitter/X, Instagram).

## Commands

### Running the Application

```pwsh
# Install dependencies (first time setup)
pip install -r requirements.txt

# Run the Streamlit app
streamlit run fake_account_detector_app.py

# Run with specific port
streamlit run fake_account_detector_app.py --server.port 8501
```

### Model Files

The application requires two model files in the `models/` directory:
- `models/cnn_model.pth` - PyTorch CNN model for feature extraction
- `models/xgb_model_tuned.json` - XGBoost classifier model

The app uses a relative path: `OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'models')`

### Environment Variables

Set these environment variables to use live API features:

```pwsh
$env:REDDIT_CLIENT_ID = "your_reddit_client_id"
$env:REDDIT_CLIENT_SECRET = "your_reddit_secret"
$env:REDDIT_USER_AGENT = "your_user_agent"
$env:TWITTER_BEARER_TOKEN = "your_twitter_bearer_token"
```

## Architecture

### Model Pipeline

1. **Text Input Processing**
   - Bio text, posts, comments, or tweets are collected
   - Text is tokenized and processed through BERT (`bert-base-uncased`)
   - BERT outputs 768-dimensional embeddings (CLS token)

2. **CNN Feature Extraction** (`CNN` class, lines 110-125)
   - Input: 768-dim BERT embeddings
   - Architecture: Conv1d → ReLU → MaxPool1d → Fully Connected
   - Output: 64-dimensional feature vector

3. **Feature Engineering** (`preprocess_input_features`, lines 204-230)
   - CNN features (64 dims)
   - Numerical metadata (9 features, log-transformed):
     - `followers_count`, `following_count`, `post_count`
     - `username_length`, `username_digit_count`
     - `mean_likes`, `mean_comments`, `mean_hashtags`, `upload_interval_std`
   - Platform one-hot encoding (2 dims: instagram, twitter)
   - Final feature vector: 64 + 9 + 2 = 75 dimensions

4. **XGBoost Classification**
   - Takes 75-dim feature vector
   - Outputs probability: [legitimate, fake]
   - Threshold: 0.5 (>0.5 = fake/automated)

### Key Functions

- `get_bert_embeddings()` (lines 128-148): Batch processes text through BERT, handles empty inputs
- `preprocess_input_features()` (lines 204-230): Combines all features into model input
- `safe_predict_proba()` (lines 232-237): Wrapper for XGBoost prediction with error handling

### API Integration

- **Reddit** (via `praw`): Fetches submissions and comments from user profile
- **Twitter** (via `tweepy`): Fetches user metadata and recent tweets
- Both use cached clients via `@st.cache_resource`

### UI Structure

- **Sidebar**: Mode selection and input controls
- **Main area**: Prediction results with confidence visualization
- **Right column**: Key metrics display
- Custom CSS styling with dark theme (lines 28-86)

## Development Notes

### GPU/CPU Detection

The app automatically detects CUDA availability:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Caching

All models and API clients use Streamlit's `@st.cache_resource` decorator to prevent reloading on every interaction.

### Model Loading Behavior

If model files are not found, the app displays warnings but doesn't crash. Predictions will fail gracefully with error messages.

### Platform Metadata Mapping

- **Reddit**: Uses `comment_karma` as followers proxy, `link_karma` as posts proxy
- **Twitter**: Standard metrics (followers, following, tweet count)
- **Manual input**: All metadata defaults to 0 (UI inputs are for display only)

### API Rate Limits

When fetching data:
- Reddit: Configurable number of submissions and comments (default 10 each)
- Twitter: Configurable number of tweets (default 20, max 100)

## File Structure

```
CapstoneProject/
├── models/
│   ├── cnn_model.pth            # CNN model weights (PyTorch)
│   └── xgb_model_tuned.json     # XGBoost classifier
├── fake_account_detector_app.py # Main Streamlit application
├── requirements.txt             # Minimal dependencies for deployment
└── WARP.md                      # This file
```

## Dependencies

Key packages:
- `streamlit==1.51.0` - Web app framework
- `torch==2.8.0+cu126` - PyTorch for CNN
- `transformers==4.57.1` - BERT model
- `xgboost==3.1.1` - Final classifier
- `praw==7.8.1` - Reddit API
- `tweepy==4.16.0` - Twitter API

Python version: **3.10.6**

## Important Implementation Details

### Text Handling
- Empty or None text inputs are replaced with `"[EMPTY]"` token before BERT processing
- BERT max sequence length: 128 tokens
- Batch size for BERT: 16

### Numerical Stability
- All numerical features undergo `log1p` transformation (log(1+x))
- Negative values are clipped to 0 before transformation

### Confidence Display
The app uses a custom confidence bar (lines 288-297) showing probability of being fake/automated.

### Error Handling
- API failures display user-friendly error messages
- Model loading failures show warnings but don't crash the app
- Prediction failures return default probabilities `[1.0, 0.0]` (legitimate)
