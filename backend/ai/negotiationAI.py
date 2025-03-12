import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import nltk
from collections import deque
from nltk.tokenize import word_tokenize

nltk.download("punkt_tab")

# ==========================
# 1. Deep Q-Network (DQN)
# ==========================

class DQNNegotiator(nn.Module):
    def __init__(self, input_size, action_size):
        super(DQNNegotiator, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)  # Outputs Q-values for each action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

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

def getReward(offeredDiscount, requestDiscount, allowedDiscount, accepted):
    reward = 0
    if accepted:
        reward += 10  # Reward for successful negotiation
    else:
        reward -= 5  # Penalty for rejection

    diff = abs(offeredDiscount - requestDiscount)
    reward -= diff / 10  # Small penalty for deviation from the request

    if offeredDiscount > allowedDiscount:
        reward -= 20  # Heavy penalty for exceeding allowed discount

    return reward

# ==========================
# 4. State Representation
# ==========================

def get_state(buyer_history, request_discount, allowed_discount):
    return np.array([
        1 if buyer_history.get('isFrequentBuyer', False) else 0,
        request_discount / 100.0,  # Normalize
        allowed_discount / 100.0
    ])

# ==========================
# 5. Action Selection (Epsilon-Greedy)
# ==========================

discount_actions = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

def choose_action(model, state, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(discount_actions)  # Explore
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = model(state_tensor)
        return discount_actions[torch.argmax(q_values).item()]  # Exploit

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
        state = get_state(buyer_history, request_discount, allowed_discount)
        discount = choose_action(model, state)
        return f"The best discount I can offer is {discount}%."
    else:
        return "Hmmmmmm."

# ==========================
# 9. Training & Running the Model
# ==========================

# Initialize Model
model = DQNNegotiator(input_size=3, action_size=len(discount_actions))
optimizer = optim.Adam(model.parameters(), lr=0.01)
buffer = ReplayBuffer()

# Simulated Training Loop
for _ in range(1000):
    buyer_history = {"isFrequentBuyer": True}
    request_discount = 15
    allowed_discount = 25

    state = get_state(buyer_history, request_discount, allowed_discount)
    action = choose_action(model, state)
    reward = getReward(action, request_discount, allowed_discount, accepted=(action <= allowed_discount))
    next_state = get_state(buyer_history, request_discount, allowed_discount)

    buffer.push(state, action, reward, next_state, done=False)
    train_dqn(model, buffer, optimizer)

# Save Model
torch.save(model.state_dict(), "negotiation_dqn.pth")

# ==========================
# 10. Testing with User Input
# ==========================

print("\nNegotiation Chatbot Ready. Type your messages below.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = respond_to_user(user_input, model, buyer_history, request_discount, allowed_discount)
    print(f"Bot: {response}")
