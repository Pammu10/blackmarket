const express = require("express");
const axios = require("axios");
const db = require("../config/db");

const router = express.Router();

const negotiationMessages = {
  positive: [
    "Well played! You’ve unlocked the best deal I can offer—enjoy your victory!",
  "You drive a hard bargain! I’m impressed. Deal sealed, and you win this round!",
  "Congratulations, you’ve cracked my discount code! But let’s keep that between us.",
"You haggled like a pro! Here’s the best price I can offer—use it wisely.",
"I respect a skilled negotiator. You’ve earned this discount. Let’s shake hands… virtually!",
"Alright, you win this time! But next time, I’ll be even sharper. Deal?",
"I ran the calculations, and it turns out… you’re good at this! Deal accepted!",
"Great deal! I’ll notify my circuits to process this before I regret it.",
"Fine, you’ve twisted my robotic arm. Enjoy the discount!",
"Consider this a reward for your persistence. But don’t tell the others!",

  ],
  neutral: [
    "This is the best I can do.",
    "A fair offer, don't you think?",
    "Let's make a deal."
  ],
  negative: [
    "Looks like we’ve hit a wall—unless you want to negotiate with my ‘no’ function?",
"I’ve checked my settings, and unfortunately, 'giving stuff away' isn’t one of them.",
"Oof, that’s a discount even my creator wouldn’t approve. Can’t do it!",
"I ran the numbers, and… nope, I’d go bankrupt. Let’s try something more realistic!",
"I’d love to make this happen, but my boss is already side-eyeing me. No can do!",
"I respect the hustle, but if I went any lower, I’d be negotiating myself out of existence.",
"I see what you’re trying to do… and I respect it. But no, final answer!",
"I’ve reached my maximum discount limit. Anything lower, and I’ll have to charge emotional damage.",
"I appreciate your determination, but if I say yes, I might get replaced by a stricter AI!",
"Let’s call it what it is—you want a deal, I want fairness. And right now, fairness wins!",
  ]
};


router.post("/", async (req, res) => {
  const { userId, productId, message } = req.body;
  if (!userId || !productId || !message) {
    return res.status(400).json({ error: "Missing required fields" });
  }

  try {
    // Fetch user and product details from the database
    const [buyerRows] = await db.execute("SELECT isFrequentBuyer FROM users WHERE id = ?", [userId]);
    const [productRows] = await db.execute("SELECT base_price, max_discount FROM products WHERE product_id = ?", [productId]);

    if (buyerRows.length === 0 || productRows.length === 0) {
      return res.status(404).json({ error: "User or product not found" });
    }

    const buyerHistory = { isFrequentBuyer: buyerRows[0].isFrequentBuyer };
    const product = productRows[0];
    const discountPerc = product.max_discount;
    console.log(parseInt(discountPerc));
    // Extract requested discount from the message
    const match = message.match(/(\d+)\s*%|\b(\d+)\s*percent\b/i);
console.log("Match:", match);

const requestDiscount = match ? parseInt(match[1] || match[2], 10) : 0;
console.log("Requested Discount:", requestDiscount);
    // Call the AI model via FastAPI
    const response = await axios.post("http://localhost:8000/negotiate/", {
      message,
      is_frequent_buyer: buyerHistory.isFrequentBuyer,
      request_discount: requestDiscount,
      allowed_discount: parseInt(discountPerc)
    });
    const {discount: finalDiscount, mood} = response.data;
    const randomMessage = negotiationMessages["negative"][Math.floor(Math.random() * negotiationMessages[mood].length)];
    console.log(finalDiscount);
    if (finalDiscount === "none") {
      return res.json({
        message: `${randomMessage}`,
        finalPrice: product.base_price,
        offerId: `${userId}-${productId}`,
        mood: mood
      });
    }
    const finalPrice = product.base_price - (product.base_price * finalDiscount / 100);

    // await db.execute(
    //   "INSERT INTO negotiations (user_id, product_id, requested_discount, offered_discount, final_price, status) VALUES (?, ?, ?, ?, ?, ?)",
    //   [userId, productId, requestDiscount, finalDiscount, finalPrice, "pending"]
    // );
    // Keep your original JSON response format
    res.json({
      message: `The best discount I can offer is ${finalDiscount}% with a final price of ${finalPrice}. ${randomMessage}`,
      finalPrice: finalPrice,
      offerId: `${userId}-${productId}`,
      mood: mood
    });

  } catch (error) {
    console.error("Negotiation error:", error);
    res.status(500).json({ error: "Negotiation failed" });
  }
});

module.exports = router;
