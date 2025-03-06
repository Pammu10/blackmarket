/*
 * rlNegotiationAI.js
 *
 * A simple Q-learning agent for price negotiation.
 * The agent learns to choose an optimal discount offer based on the buyer's history and request.
 */

const fs = require("fs");

class QLearningNegotiator {
  constructor(options) {
    this.learningRate = options.learningRate || 0.1;
    this.discountFactor = options.discountFactor || 0.9;
    this.epsilon = options.epsilon || 0.2; // exploration rate
    this.actions = options.actions || [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]; // possible discount percentages
    // Q table structure: { state: { action: Q_value, ... } }
    this.qTable = {};
  }

  // Define state based on buyer history and requested discount.
  // Here we use:
  //   - buyerState: 1 if frequent buyer, 0 otherwise.
  //   - discountBucket: "low" (<10%), "medium" (10-20%), or "high" (>=20%)
  getState(buyerHistory, requestDiscount) {
    const buyerState = buyerHistory && buyerHistory.isFrequentBuyer ? 1 : 0;
    let discountBucket = "";
    if (requestDiscount < 10) discountBucket = "low";
    else if (requestDiscount < 20) discountBucket = "medium";
    else discountBucket = "high";
    return `${buyerState}-${discountBucket}`;
  }

  // Initialize the Q-table entry for a state if it doesn't exist.
  initializeState(state) {
    if (!this.qTable[state]) {
      this.qTable[state] = {};
      for (let action of this.actions) {
        this.qTable[state][action] = 0; // start with 0 value for all actions
      }
    }
  }

  // Epsilon-greedy action selection: explore or exploit.
  chooseAction(state) {
    this.initializeState(state);
    if (Math.random() < this.epsilon) {
      // Exploration: choose a random action
      const randomIndex = Math.floor(Math.random() * this.actions.length);
      return this.actions[randomIndex];
    } else {
      // Exploitation: choose the action with the highest Q value
      let maxQ = -Infinity;
      let bestAction = this.actions[0];
      for (let action of this.actions) {
        if (this.qTable[state][action] > maxQ) {
          maxQ = this.qTable[state][action];
          bestAction = action;
        }
      }
      return bestAction;
    }
  }

  // Q-learning update rule.
  updateQ(state, action, reward, nextState) {
    this.initializeState(nextState);
    const currentQ = this.qTable[state][action];
    // Find the maximum Q value for the next state.
    let maxNextQ = Math.max(...this.actions.map(a => this.qTable[nextState][a]));
    // Update Q value.
    this.qTable[state][action] = currentQ + this.learningRate * (reward + this.discountFactor * maxNextQ - currentQ);
  }

  // Save Q-table to a file (optional).
  saveQTable(filePath) {
    fs.writeFileSync(filePath, JSON.stringify(this.qTable, null, 2));
  }

  // Load Q-table from a file (optional).
  loadQTable(filePath) {
    if (fs.existsSync(filePath)) {
      this.qTable = JSON.parse(fs.readFileSync(filePath));
    }
  }
}

// A sample reward function to evaluate the offered discount.
// It gives positive rewards for acceptance and penalizes deviations from the request.
// If the offered discount exceeds the allowed discount, it heavily penalizes.
function getReward(offeredDiscount, requestDiscount, allowedDiscount, accepted) {
  let reward = 0;
  if (accepted) {
    reward += 10; // reward for successful negotiation
  } else {
    reward -= 5; // penalty for rejection
  }
  const diff = Math.abs(offeredDiscount - requestDiscount);
  reward -= diff / 10; // penalty for offering a discount far from the request
  if (offeredDiscount > allowedDiscount) {
    reward -= 20; // heavy penalty for exceeding allowed discount
  }
  return reward;
}

module.exports = {
  QLearningNegotiator,
  getReward
};
