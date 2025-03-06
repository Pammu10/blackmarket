// backend/routes/rlNegotiations.js

const express = require("express");
const db = require("../config/db"); // MySQL connection (ensure you have this file)
const { QLearningNegotiator, getReward } = require("../ai/rlNegotiationAI");
const router = express.Router();

// Initialize the RL negotiator with desired parameters.
const negotiator = new QLearningNegotiator({
  learningRate: 0.1,
  discountFactor: 0.9,
  epsilon: 0.2,
  actions: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
});

// POST /qlearn
// This route receives a buyer's negotiation request including the buyer ID, product ID, and a message.
router.post("/", async (req, res) => {
  /* Expected request body:
    {
      "userId": 1,                 // ID of the buyer (from the users table)
      "productId": 1,              // ID of the product (from the products table)
      "message": "Can I get 20% off?"
    }
  */
  const { userId, productId, message } = req.body;
  if (!userId || !productId || !message) {
    return res.status(400).json({ error: "Missing required fields: userId, productId, or message" });
  }
  
  try {
    // 1. Retrieve buyer details to determine if they are a frequent buyer.
    const [buyerRows] = await db.execute("SELECT isFrequentBuyer FROM users WHERE id = ?", [userId]);
    if (buyerRows.length === 0) {
      return res.status(404).json({ error: "Buyer not found" });
    }
    const buyerHistory = { isFrequentBuyer: buyerRows[0].isFrequentBuyer };

    // 2. Retrieve product details to get basePrice and allowedDiscount.
    const [productRows] = await db.execute("SELECT price, allowedDiscount FROM products WHERE id = ?", [productId]);
    if (productRows.length === 0) {
      return res.status(404).json({ error: "Product not found" });
    }
    const product = productRows[0];

    // 3. Extract requested discount from the message (e.g., "Can I get 20% off?")
    const discountRegex = /(\d+)%/;
    let requestDiscount = 0;
    const match = message.match(discountRegex);
    if (match) {
      requestDiscount = parseInt(match[1], 10);
    } else {
      // If no discount is specified, assume 0.
      requestDiscount = 0;
    }

    // 4. Build the negotiation request object.
    const negotiationRequest = {
      message,
      basePrice: product.price,
      buyerHistory,
      requestDiscount,
      allowedDiscount: product.allowedDiscount
    };

    // 5. Use the RL negotiator.
    const state = negotiator.getState(buyerHistory, requestDiscount);
    const offeredDiscount = negotiator.chooseAction(state);
    // For this example, we consider the offer accepted if it is greater than or equal to the requested discount.
    const accepted = offeredDiscount >= requestDiscount;
    const reward = getReward(offeredDiscount, requestDiscount, product.allowedDiscount, accepted);
    // For simplicity, we use the same state as the next state.
    negotiator.updateQ(state, offeredDiscount, reward, state);
    const finalPrice = product.price * (1 - offeredDiscount / 100);

    res.json({
      offeredDiscount,
      finalPrice,
      accepted,
      reward,
      message: `Our RL agent offers a discount of ${offeredDiscount}%.`
    });
  } catch (error) {
    console.error("Negotiation error:", error);
    res.status(500).json({ error: "Negotiation failed" });
  }
});

module.exports = router;
