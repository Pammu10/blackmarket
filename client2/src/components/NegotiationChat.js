import React, { useEffect, useState } from "react";
import { fetchProducts, negotiatePrice } from "../services/api";

const NegotiationChat = () => {
  const [message, setMessage] = useState("");
  const [selectedProduct, setSelectedProduct] = useState("");
  const [products, setProducts] = useState([]);
  const [responses, setResponses] = useState([]);

  const handleSend = async () => {
    if (!message.trim() || !selectedProduct) return;
    const userId = 1;
    try {
      const response = await negotiatePrice(message, selectedProduct, userId);
      setResponses([...responses, { user: message, bot: response.message , finalPrice: response.finalPrice }]);
      setMessage("");
    } catch(error) {
      console.error(error);
    }
    
    // const {offeredDiscount, finalPrice, accepted, reward, message} = response;
    
  };
  useEffect(() => {
    const getProducts = async () => {
      try {
        const productsList = await fetchProducts();
        setProducts(productsList);
        if (productsList.length > 0) {
          setSelectedProduct(productsList[0].name);
        }
      } catch (error) {
        console.error("Failed to fetch products", error);
      }
    };getProducts();
  }, []);
  return (
    <div className="chat-box">
      <h2>💬 Negotiate Your Price</h2>
      
      <div className="product-selector">
        <label htmlFor="product-select">Select Product:</label>
        <select
          id="product-select"
          value={selectedProduct}
          onChange={(e) => setSelectedProduct(e.target.value)}
        >
          {products.map((product) => (
            <option key={product.product_id} value={product.product_id}>
              {product.name} (${product.base_price})
            </option>
          ))}
        </select>
        <img src={selectedProduct.image_url} alt={selectedProduct.name} width={300}/>
      </div>
      
      <div className="chat-history">
        {responses.map((res, index) => (
          <div key={index}>
            <p><strong>You:</strong> {res.user}</p>
            <p><strong>AI:</strong> {res.bot} with a final price of  {res.finalPrice}</p>
          </div>
        ))}
      </div>
      
      <div className="chat-input">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Ask for a discount..."
        />
        <button onClick={handleSend}>Send</button>
      </div>
    </div>

  );
};

export default NegotiationChat;
