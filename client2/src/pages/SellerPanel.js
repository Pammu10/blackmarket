import React, { useState } from "react";
import { setNegotiationRule } from "../services/api";

const SellerPanel = () => {
  const [minDiscount, setMinDiscount] = useState(5);
  const [maxDiscount, setMaxDiscount] = useState(30);

  const handleSubmit = async () => {
    await setNegotiationRule({ minDiscount, maxDiscount });
    alert("✅ Negotiation rules updated!");
  };

  return (
    <div className="seller-panel">
      <h1>⚙️ Seller Panel</h1>
      <label>Minimum Discount (%)</label>
      <input 
        type="number" 
        value={minDiscount} 
        onChange={(e) => setMinDiscount(e.target.value)}
      />
      
      <label>Maximum Discount (%)</label>
      <input 
        type="number" 
        value={maxDiscount} 
        onChange={(e) => setMaxDiscount(e.target.value)}
      />

      <button onClick={handleSubmit}>Save Rules</button>
    </div>
  );
};

export default SellerPanel;
