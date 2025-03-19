import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
import random

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOActorCritic, self).__init__()
        
        # Actor (Policy Network)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)  # Outputs probabilities for actions
        )

        # Critic (Value Network)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Outputs single value estimation
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

    def select_action(self, state, temperature=0.5):
        with torch.no_grad():
            action_probs, _ = self.forward(state)
        scaled_probs = torch.softmax(action_probs / temperature, dim=-1)
        action = torch.multinomial(scaled_probs, 1).item()
        return action, scaled_probs[:, action]

class PPOReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def store(self, state, action, log_prob, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def clear(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.next_states, self.dones = [], [], []

def compute_ppo_loss(actor_critic, buffer, gamma=0.99, clip_ratio=0.2):
    states = torch.tensor(buffer.states, dtype=torch.float32)
    actions = torch.tensor(buffer.actions, dtype=torch.int64)
    log_probs_old = torch.tensor(buffer.log_probs, dtype=torch.float32)
    rewards = torch.tensor(buffer.rewards, dtype=torch.float32)
    dones = torch.tensor(buffer.dones, dtype=torch.float32)

    _, values = actor_critic(states)
    values = values.squeeze()
    advantages = rewards + gamma * (1 - dones) * values - values.detach()

    new_action_probs, new_values = actor_critic(states)
    new_log_probs = torch.log(new_action_probs.gather(1, actions.unsqueeze(1)).squeeze(1))
    
    ratio = torch.exp(new_log_probs - log_probs_old)
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    value_loss = nn.MSELoss()(new_values.squeeze(), rewards)
    entropy_bonus = -0.05 * torch.sum(new_action_probs * torch.log(new_action_probs + 1e-8), dim=1).mean()

    return policy_loss + 0.5 * value_loss + entropy_bonus

def train_ppo(actor_critic, buffer, optimizer, epochs=10):
    for _ in range(epochs):
        loss = compute_ppo_loss(actor_critic, buffer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    buffer.clear()

sia = SentimentIntensityAnalyzer()
intent_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_mood_score(text):
    scores = sia.polarity_scores(text)
    if scores["compound"] >= 0.1:
        return "positive"
    elif scores["compound"] <= -0.1:
        return "negative"
    else:
        return "neutral"

def get_intent_embedding(text):
    return intent_model.encode(text)

def get_state(buyer_history, request_discount, allowed_discount, mood_score, intent_embedding):
    mood_map = {"negative": -1, "neutral": 0, "positive": 1}
    return np.concatenate((
        np.array([
            1 if buyer_history.get('isFrequentBuyer', False) else 0,
            request_discount / 100.0,
            allowed_discount / 100.0,
            mood_map[mood_score]
        ]),
        intent_embedding
    ))

def getReward(discount, request_discount, allowed_discount, accepted, mood_score):
    mood_rewards = {"negative": -2, "neutral": 0, "positive": 2}
    reward = 20 - 2 * abs(request_discount - discount) if accepted else -10
    if discount > allowed_discount:
        reward -= 5
    reward += mood_rewards[mood_score]
    return reward

discount_actions = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

def negotiate(user_input, model, buffer, buyer_history, request_discount, allowed_discount):
    mood_score = get_mood_score(user_input)
    intent_embedding = get_intent_embedding(user_input)
    state = get_state(buyer_history, request_discount, allowed_discount, mood_score, intent_embedding)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    action, log_prob = model.select_action(state_tensor)
    discount = discount_actions[action]
    accepted = discount <= allowed_discount
    reward = getReward(discount, request_discount, allowed_discount, accepted, mood_score)
    next_state = get_state(buyer_history, request_discount, allowed_discount, mood_score, intent_embedding)

    buffer.store(state, action, log_prob.item(), reward, next_state, accepted)
    return discount, accepted

actor_critic = PPOActorCritic(state_dim=388, action_dim=len(discount_actions))
optimizer = optim.Adam(actor_critic.parameters(), lr=0.001)
buffer = PPOReplayBuffer()

sample_inputs = ["Can I get a better deal?", "Give me a discount.", "I'm a regular customer; you should treat me better."]

for episode in range(10000):
    user_input = random.choice(sample_inputs)
    buyer_history = {"isFrequentBuyer": random.choice([True, False])}
    request_discount = random.choice(discount_actions)
    allowed_discount = random.choice(discount_actions)
    print(f"Allowed Discount: {allowed_discount}% Requested Discount: {request_discount}%")
    discount, accepted = negotiate(user_input, actor_critic, buffer, buyer_history, request_discount, allowed_discount)
    if accepted:
        print(f"Episode {episode}: Offer {discount}% - Accepted ✅")
    else:
        print(f"Episode {episode}: Offer {discount}% - Rejected ❌")
    train_ppo(actor_critic, buffer, optimizer)

torch.save(actor_critic.state_dict(), "ppo_negotiation.pth")
print("PPO Model Saved!")
