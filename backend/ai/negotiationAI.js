/*
 * negotiationAI.js
 *
 * This module implements a simple, rule-based AI negotiation logic for our eCommerce app.
 * It extracts the requested discount from the buyer's message, then compares it against
 * seller-defined thresholds and buyer history. Based on these factors, it either accepts the request
 * or proposes a counter discount.
 */

function negotiatePrice(request) {
    // 'request' is an object that should include:
    //  - message: the buyer's message (e.g., "Can I get 20% off?")
    //  - basePrice: the original price of the product
    //  - buyerHistory: an object, e.g., { isFrequentBuyer: true }
    //
    // Seller rules (could come from the DB) are defined as:
    const sellerRules = { minDiscount: 5, maxDiscount: 50 };
  
    // Extract requested discount from the message using regex.
    const discountRegex = /(\d+)%/;
    let requestedDiscount = 0;
    const match = request.message.match(discountRegex);
    if (match) {
      requestedDiscount = parseInt(match[1], 10);
    }
  
    // Calculate allowed discount based on seller rules and buyer history.
    let allowedDiscount = sellerRules.minDiscount;
    if (request.buyerHistory && request.buyerHistory.isFrequentBuyer) {
      allowedDiscount += 10; // Give an extra 10% for loyal buyers.
    }
    // Ensure allowedDiscount does not exceed seller's maximum allowed discount.
    if (allowedDiscount > sellerRules.maxDiscount) {
      allowedDiscount = sellerRules.maxDiscount;
    }
  
    // Determine negotiation outcome.
    let accepted = false;
    let finalDiscount = 0;
    let responseMessage = "";
  
    if (requestedDiscount <= allowedDiscount) {
      accepted = true;
      finalDiscount = requestedDiscount;
      responseMessage = `Your request for ${requestedDiscount}% discount is accepted.`;
    } else {
      accepted = false;
      finalDiscount = allowedDiscount;
      responseMessage = `We cannot offer ${requestedDiscount}% off, but we can offer a ${allowedDiscount}% discount instead.`;
    }
  
    // Calculate the final price after discount.
    const finalPrice = request.basePrice * (1 - finalDiscount / 100);
  
    return {
      accepted,
      finalDiscount,
      finalPrice,
      responseMessage,
    };
  }
  
  module.exports = {
    negotiatePrice,
  };
  