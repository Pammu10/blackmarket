# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer
# from fastapi import FastAPI
# from pydantic import BaseModel
# from nltk.tokenize import word_tokenize


# # FastAPI Setup
# app = FastAPI()

# # Load Sentiment and Intent Models
# sentiment_pipeline = pipeline("sentiment-analysis")
# intent_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# class NegotiationRequest(BaseModel):
#     message: str
#     is_frequent_buyer: bool
#     request_discount: int
#     allowed_discount: int

# # Discount Levels
# discount_actions = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# # PPO Actor-Critic Model
# class PPOActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(PPOActorCritic, self).__init__()
#         self.actor = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, action_dim),
#             nn.Softmax(dim=-1)
#         )
#         self.critic = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#     def forward(self, state):
#         action_probs = self.actor(state)
#         state_value = self.critic(state)
#         return action_probs, state_value

#     def select_action(self, state):
#         with torch.no_grad():
#             action_probs, _ = self.forward(state)
#         action = torch.multinomial(action_probs, 1).item()
#         return action, action_probs[:, action]

# # Load PPO Model
# actor_critic = PPOActorCritic(state_dim=5, action_dim=len(discount_actions))
# actor_critic.load_state_dict(torch.load("ppo_negotiation.pth"))
# actor_critic.eval()

# # Function to get sentiment
# def get_mood_score(text):
#     result = sentiment_pipeline(text)[0]["label"]
#     return {
#         "POSITIVE": "positive",
#         "NEGATIVE": "negative",
#         "NEUTRAL": "neutral"
#     }.get(result, "neutral")

# # Function to get intent
# def get_intent(text):
#     intents = [
#         "polite_request", "direct_demand", "aggressive_demand", 
#         "manipulative_tactic", "psychological_pressure"
#     ]
#     intent_embeddings = intent_model.encode(intents)
#     text_embedding = intent_model.encode([text])
#     scores = np.dot(intent_embeddings, text_embedding.T).squeeze()
#     return intents[np.argmax(scores)]

# # State Representation
# mood_map = {"negative": -1, "neutral": 0, "positive": 1}
# intent_map = {
#     "polite_request": 1, "direct_demand": 0,
#     "aggressive_demand": -1, "manipulative_tactic": -0.5,
#     "psychological_pressure": -0.5
# }

# def get_state(is_frequent_buyer, request_discount, allowed_discount, mood_score, intent):
#     return np.array([
#         1 if is_frequent_buyer else 0,
#         request_discount / 100.0,
#         allowed_discount / 100.0,
#         mood_map[mood_score],
#         intent_map[intent]
#     ])

# # Reward Function
# def getReward(discount, request_discount, allowed_discount, accepted, mood_score, intent):
#     reward = 10 - abs(request_discount - discount) if accepted else -5
#     if discount > allowed_discount:
#         reward -= 2
#     reward += mood_map[mood_score] * 2  # Boost polite buyers, penalize aggressive ones
#     reward += intent_map[intent] * 2
#     return reward

# def is_discount_request(user_input):
#     keywords = ["discount", "offer", "lower price", "best price", "deal"]
#     tokens = word_tokenize(user_input.lower())
#     return any(word in tokens for word in keywords)

# # Negotiation Endpoint
# @app.post("/negotiate")
# def negotiate(request: NegotiationRequest):
#     mood_score = get_mood_score(request.message)
#     if not is_discount_request(request.message):
#         return {"discount": "none", "mood": mood_score}
#     intent = get_intent(request.message)
#     state = get_state(request.is_frequent_buyer, request.request_discount, request.allowed_discount, mood_score, intent)
#     state_tensor = torch.FloatTensor(state).unsqueeze(0)
#     action, _ = actor_critic.select_action(state_tensor)
#     discount = discount_actions[action]
#     accepted = discount <= request.allowed_discount
#     return {"discount": discount, "accepted": accepted, "mood": mood_score, "intent": intent}






from fastapi import FastAPI
import torch
import torch.nn as nn
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from pydantic import BaseModel
from nltk.tokenize import word_tokenize

# Initialize FastAPI
app = FastAPI()

# Initialize sentiment analyzer
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Define discount actions
discount_actions = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# Deep Q-Network (DQN) Model
class DQNNegotiator(nn.Module):
    def __init__(self, input_size, action_size):
        super(DQNNegotiator, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, action_size)
        self.dropout = nn.Dropout(0.2)  # Prevent overfitting

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)
# Load Model
def load_model():
    model = DQNNegotiator(input_size=4, action_size=len(discount_actions))
    model.load_state_dict(torch.load("negotiation_dqn.pth", map_location=torch.device('cpu')))  # Ensure compatibility
    model.eval()  # Set to evaluation mode to avoid BatchNorm issues
    return model

model = load_model()

# Define API request model
class NegotiationRequest(BaseModel):
    message: str
    is_frequent_buyer: bool
    request_discount: int
    allowed_discount: int

# Function to analyze mood
def predict_mood(text):
    scores = sia.polarity_scores(text)
    if scores["compound"] >= 0.1:
        return "positive"
    elif scores["compound"] <= -0.1:
        return "negative"
    else:
        return "neutral"

# Function to get state for model
def get_state(buyer_history, request_discount, allowed_discount, mood):
    mood_map = {"negative": -1, "neutral": 0, "positive": 1}
    return np.array([
        1 if buyer_history else 0,
        request_discount / 100.0,
        allowed_discount / 100.0,
        mood_map[mood]  # Convert mood to numerical value
    ])
def is_discount_request(user_input):
    keywords = ["discount", "offer", "lower price", "best price", "deal"]
    tokens = word_tokenize(user_input.lower())
    return any(word in tokens for word in keywords)


# Define negotiation API route
@app.post("/negotiate")
def negotiate(req: NegotiationRequest):
    print(req)
    if is_discount_request(req.message):
        mood = predict_mood(req.message)
        state = get_state(req.is_frequent_buyer, req.request_discount, req.allowed_discount, mood)

        # Choose action
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = model(state_tensor)
            action = discount_actions[torch.argmax(q_values).item()]

        return {"discount": action, "mood": mood}
    else:
        mood = predict_mood(req.message)
        return {"discount": "none", "mood": "neutral"}
