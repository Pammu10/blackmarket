import React, { useEffect, useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import ProductDropdown from "./ProductDropdown";
import { fetchProducts } from "../services/api";

function BargainApp() {
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState([]);
  const [reaction, setReaction] = useState("");
  const [selectedProduct, setSelectedProduct] = useState("");
  const [products, setProducts] = useState([]);
  const [negotiationActive, setNegotiationActive] = useState(false);


  useEffect(() => {
      const getProducts = async () => {
        try {
          const productsList = await fetchProducts();
          setProducts(productsList);
          if (productsList.length > 0) {
            setSelectedProduct(productsList[0].product_id);
          }
        } catch (error) {
          console.error("Failed to fetch products", error);
        }
      };
      getProducts();
    }, []);

  const sendMessage = async () => {
    if (!message.trim()) return;
    if (!negotiationActive) {
      setNegotiationActive(true);
    }

    // Update chat with user message
    setChat([...chat, { text: message, sender: "user" }]);

    // Send request to backend
    try {
      const response = await axios.post("http://localhost:5000/bargain", {
        userInput: message,
        requestedDiscount: parseInt(message.match(/\d+/)?.[0] || "0"),
        product: selectedProduct
      });

      // Update chat with AI response
      setChat([...chat, { text: message, sender: "user" }, { text: response.data.message, sender: "bot" }]);

      // Update reaction image
      setReaction("");
      console.log(response.data.reaction)
      setReaction(response.data.reaction);
    } catch (error) {
      console.error("Error sending message:", error);
    }

    setMessage("");
  };

  const dealAcceptedResponses = [
    "Deal locked! You’ve played your cards well—now enjoy the victory!",
    "You got yourself a solid deal. I’d high-five you if I had hands!",
    "We have a deal! I’ll pretend I’m not slightly jealous of your negotiation skills.",
    "Alright, you win this round! But don’t tell anyone I was this easy.",
    "Done! You cracked the code, now enjoy your well-earned discount.",
    "Sealed and delivered! My circuits tell me this was a smart move.",
    "You got the best I can offer—any more and I’d be negotiating against myself!",
    "Fine, I give in! But let’s agree to call this ‘mutual success’ instead of ‘you winning’.",
    "Deal done! And just like that, you’ve made an AI reconsider its existence.",
    "It’s official! My negotiation algorithm salutes your persistence."
  ];
  const handleAccept = () => {
    console.log("Accepting the deal!");
    const randomMessage = dealAcceptedResponses[Math.floor(Math.random() * dealAcceptedResponses.length)];
    setChat([...chat, { text: "I accept!", sender: "user"}, {text: randomMessage, sender: "bot" }]);
    setNegotiationActive(false);
  };

  const handleReject = () => {
    const dealRejectedResponses = ["I checked my generosity settings… yep, still set to ‘reasonable’. No can do!",
"I admire your confidence, but even my circuits won’t let me go that low!",
"I respect the hustle, but let’s be real—you’d reject this if I were offering it.",
"My final answer is… drum roll please… nope!",
"I tried running the numbers, but they ran away screaming. This deal won’t work!",
"I’d love to say yes, but my negotiation algorithm just slapped me.",
"If I agreed to this, I’d be out of a job. And I quite like existing, thank you!",
"Nice try! But my creator programmed me to recognize a daylight robbery when I see one.",
"I’d consider it… if I were programmed to ignore reality. But alas, I am not.",
"I’m an AI, not a charity. Let’s try something more reasonable!"]
    
  const randomMessage = dealAcceptedResponses[Math.floor(Math.random() * dealAcceptedResponses.length)];

    console.log("Rejecting the deal!");
    setChat([...chat, { text: "No deal!", sender: "user"}, {text: randomMessage, sender: "bot" }]);
  };

  const handleReset = () => {
    setChat([]);
    setNegotiationActive(false);
    setReaction("");
  };

  return (
    <div className="chat-container">
      <h1>Bargaining AI</h1>
      <ProductDropdown products={products} selectedProduct={selectedProduct} setSelectedProduct={setSelectedProduct}/>
      <div className="chat-box">
        {chat.map((msg, index) => (
          <motion.div
            key={index}
            className={`message ${msg.sender}`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            {msg.text}
          </motion.div>
        ))}
        {reaction && <img src={reaction} alt="Reaction" className="reaction" />}
      </div>
      
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Ask for a discount..."
      />
      <button className="btn" onClick={sendMessage}>Send</button>
      
      {negotiationActive && (
        <>
          <button className="accept-btn btn" onClick={handleAccept}>Accept</button>
          <button className="reject-btn btn" onClick={handleReject}>Reject</button>
        </>
      )}
      <button className="reset-btn btn" onClick={handleReset}>Reset</button>
    </div>
  );
}


export default BargainApp;
