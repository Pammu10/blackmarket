import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import nltk
from collections import deque
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
writer = SummaryWriter("runs/negotiation_dqn")



nltk.download("punkt_tab")
nltk.download("vader_lexicon")


# ==========================
# 1. Deep Q-Network (DQN)
# ==========================

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
# ==========================
# 2. Experience Replay
# ==========================

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=32):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# ==========================
# 3. Reward Function
# ==========================

def getReward(offeredDiscount, requestDiscount, allowedDiscount, accepted, mood_score, round_number, max_rounds=5):
    """
    - Rewards gradual discount increases.
    - Penalizes offering max discount too soon.
    """
    reward = 0

    # Scaling factor based on negotiation progress
    alpha = round_number / max_rounds  

    # Encourage small increments
    if offeredDiscount <= (allowedDiscount * alpha):
        reward += 5  # Encourages gradual increase
    elif offeredDiscount < allowedDiscount:
        reward += 2  # Neutral reward
    else:
        reward -= 5  # Heavy penalty for exceeding allowedDiscount too soon

    # Reward successful negotiations
    if accepted:
        reward += 20  

        # Additional profit-based reward
        profit_margin_bonus = (allowedDiscount - offeredDiscount) / 10
        reward += profit_margin_bonus  
    else:
        reward -= 10  # Penalty for failure

    # Distance penalty from request
    distance_penalty = abs(offeredDiscount - requestDiscount) / 5
    reward -= distance_penalty

    # **Mood-based adjustments**
    if mood_score == 1:  
        reward += 5  # Reward positive sentiment
    elif mood_score == -1:  
        reward -= 5  # Penalize negative sentiment

    return reward



# ==========================
# 4. State Representation
# ==========================
sia = SentimentIntensityAnalyzer()

def get_mood_score(text):
    scores = sia.polarity_scores(text)
    if scores["compound"] >= 0.1:
        return "positive"
    elif scores["compound"] <= -0.1:
        return "negative"
    else:
        return "neutral"

def get_state(buyer_history, request_discount, allowed_discount, mood_score):
    mood_map = {"negative": -1, "neutral": 0, "positive": 1}
    return np.array([
        1 if buyer_history.get('isFrequentBuyer', False) else 0,
        request_discount / 100.0,  # Normalize
        allowed_discount / 100.0,
        mood_map[mood_score]
    ])

# ==========================
# 5. Action Selection (Epsilon-Greedy)
# ==========================

discount_actions = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

def choose_action(model, state, epsilon, round_number, max_rounds=5):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(discount_actions)))  # Exploration
    
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    q_values = model(state_tensor)
    
    # Scaling factor (alpha) increases as negotiation progresses
    alpha = min(1.0, round_number / max_rounds)  

    # Get sorted discount indices (least discount first)
    sorted_indices = torch.argsort(q_values, descending=False)
    
    # Apply alpha-based scaling: 
    # Start with the lowest discount and move up as bargaining continues
    best_index = sorted_indices[int(alpha * (len(sorted_indices) - 1))].item()

    return best_index


# ==========================
# 6. Training Function
# ==========================

def train_dqn(model, buffer, optimizer, batch_size=32, gamma=0.9):
    if buffer.size() < batch_size:
        return  # Wait until enough samples are available

    batch = buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert list of numpy arrays to a single numpy array before converting to tensor
    states = torch.tensor(np.array(states), dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)

    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Convert actions from values to indices
    action_indices = torch.tensor([discount_actions.index(a) for a in actions], dtype=torch.int64).unsqueeze(1)

    # Compute Q-values for selected actions
    q_values = model(states).gather(1, action_indices).squeeze()

    # Compute max Q-value for next states
    next_q_values = model(next_states).max(1)[0]
    expected_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Compute loss and update the model
    loss = nn.MSELoss()(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ==========================
# 7. NLP-Based Intent Recognition
# ==========================

def is_discount_request(user_input):
    keywords = ["discount", "offer", "lower price", "best price", "deal"]
    tokens = word_tokenize(user_input.lower())
    return any(word in tokens for word in keywords)

# ==========================
# 8. User Interaction Function
# ==========================

def respond_to_user(user_input, model, buyer_history, request_discount, allowed_discount):
    if is_discount_request(user_input):
        mood_score = get_mood_score(user_input)  # Extract mood score
        state = get_state(buyer_history, request_discount, allowed_discount, mood_score)
        discount = choose_action(model, state)
        return discount
    else:
        return "Hmmmmmm."

# ==========================
# 9. Training & Running the Model
# ==========================

# Initialize Model
model = DQNNegotiator(input_size=4, action_size=len(discount_actions))
optimizer = optim.Adam(model.parameters(), lr=0.01)
buffer = ReplayBuffer()
reward_history = []
epsilon_history = []

epsilon = 1.0  # Start with full exploration
epsilon_min = 0.1  # Minimum randomness
epsilon_decay = 0.995  # Decay factor
sample_inputs = [
    # **Polite Requests**
    "Can I get a better deal?",  
    "What’s the best price you can offer?",  
    "Any chance of a discount on this?",  
    "I’d really appreciate a lower price.",  

    # **Direct Demands**  
    "Give me a discount.",  
    "Lower the price, or I’m out.",  
    "I know you can do better than this.",  
    "Match the competitor’s price or I’ll buy from them.",  

    # **Rude/Confrontational**  
    "This price is ridiculous. Cut it down.",  
    "Don’t waste my time. Give me your lowest price.",  
    "Are you seriously charging this much?",  
    "I’m not paying a penny over [X]—take it or leave it.",  

    # **Manipulative Tactics**  
    "I’m a regular customer; you should treat me better.",  
    "I saw someone else get a better deal, why not me?",  
    "If I buy two, what’s the price?",  
    "I’ll post a good review if you give me a discount.",  

    # **Sarcasm/Psychological Pressure**  
    "Wow, this is the *best* deal ever… Not.",  
    "Is this price a joke? I hope it is.",  
    "I didn’t know robbery was legal here.",  
    "You must *really* want to lose a customer with this price."  
]
# Simulated Training Loop
for episode in range(10000):
    
    user_input = random.choice(sample_inputs)
    mood_score = get_mood_score(user_input)

    # Simulate different buyer types
    buyer_history = {"isFrequentBuyer": random.choice([True, False])}
    request_discount = random.randint(5, 30)  # Random requested discount (5-30%)
    allowed_discount = random.randint(10, 40)  # Max discount range

    # Get initial state
    state = get_state(buyer_history, request_discount, allowed_discount, mood_score)

    # Choose action with exploration-exploitation tradeoff
    round_number = 1  # Start negotiation round tracking
    max_rounds = 5  # Define max bargaining rounds

    while round_number <= max_rounds:
        action = choose_action(model, state, epsilon, round_number, max_rounds)
        offeredDiscount = discount_actions[action]

        accepted = offeredDiscount <= allowed_discount
        reward = getReward(offeredDiscount, request_discount, allowed_discount, accepted, mood_score, round_number, max_rounds)

        next_state = get_state(buyer_history, request_discount, allowed_discount, mood_score)

        buffer.push(state, offeredDiscount, reward, next_state, done=accepted)

        train_dqn(model, buffer, optimizer)

        writer.add_scalar("Reward", reward, episode)
        writer.add_scalar("Epsilon", epsilon, episode)

        reward_history.append(reward)
        epsilon_history.append(epsilon)

        if accepted:
            break  # End negotiation if accepted
        
        round_number += 1  # Move to next round

    # Decay epsilon to reduce exploration over time
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Track progress every 100 episodes
    if episode % 100 == 0:
        print(f"Episode {episode}: Action {action}, Reward {reward}, Epsilon {epsilon:.4f}")

writer.close()
plt.figure(figsize=(10, 5))
plt.plot(reward_history, label="Reward per Episode", color="blue")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Negotiation Model Training Performance")
plt.legend()
plt.grid()
plt.show()

# Save Model
torch.save(model.state_dict(), "negotiation_dqn.pth")
print("model saved")
# scripted_model = torch.jit.script(model)
# scripted_model.save("ai/negotiation_dqn_scripted.pth")

# print("Model saved as TorchScript!")


# ==========================
# 10. Testing with User Input
# ==========================

# print("\nNegotiation Chatbot Ready. Type your messages below.")

# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["exit", "quit"]:
#         break
#     response = respond_to_user(user_input, model, buyer_history, request_discount, allowed_discount)
#     print(f"Bot: {response}")


# def load_model():
#     input_size = 3
#     action_size = len(discount_actions)
#     model = DQNNegotiator(input_size, action_size)
#     model.load_state_dict(torch.load("ai/negotiation_dqn.pth"))
#     model.eval()
#     return model



# if __name__ == "__main__":
#     import sys
#     model = load_model()
#     user_input = sys.argv[1]
#     is_frequent_buyer = int(sys.argv[2])
#     request_discount = int(sys.argv[3])
#     allowed_discount = int(sys.argv[4])

#     buyer_history = {"isFrequentBuyer": bool(is_frequent_buyer)}
#     response = respond_to_user(user_input, model, buyer_history, request_discount, allowed_discount)

#     print(response)