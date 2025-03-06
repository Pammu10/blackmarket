// backend/routes/negotiationRoutes.js
const express = require('express');
const { authenticateUser, authorizeRole } = require('../middleware/authMiddleware');
const negotiationAI = require("../ai/negotiationAI");
const router = express.Router();

// Buyer requests a price negotiation
router.post('/request', authenticateUser, async (req, res) => {
    const { sellerId, offeredPrice } = req.body;
    const rules = await NegotiationRule.findOne({ sellerId });
    
    if (!rules) return res.status(404).json({ message: 'Seller negotiation rules not found' });
    
    let discount = 0;
    if (offeredPrice < rules.maxDiscount) {
        discount = Math.min(rules.maxDiscount, offeredPrice * 0.1);
    }
    
    res.json({ finalPrice: offeredPrice - discount, discount });
});

// Seller sets negotiation rules
router.post('/rules', authenticateUser, authorizeRole('seller'), async (req, res) => {
    const { minDiscount, maxDiscount, stockThreshold, competitorMatching } = req.body;
    
    if (maxDiscount > 50) return res.status(400).json({ message: 'Max discount cannot exceed 50%' });
    
    const rule = new NegotiationRule({
        sellerId: req.user._id,
        minDiscount,
        maxDiscount,
        stockThreshold,
        competitorMatching
    });
    await rule.save();
    res.json({ message: 'Negotiation rules set successfully' });
});


// POST /api/negotiate
router.post("/", async (req, res) => {
  // Expect the request body to include:
  // - message: the negotiation message from the buyer
  // - basePrice: the original product price
  // - buyerHistory: buyer's info, e.g., { isFrequentBuyer: true }
  const negotiationRequest = req.body;

  try {
    const result = negotiationAI.negotiatePrice(negotiationRequest);
    res.json(result);
  } catch (error) {
    console.error("Negotiation error:", error);
    res.status(500).json({ error: "Negotiation failed" });
  }
});

module.exports = router;


module.exports = router;
