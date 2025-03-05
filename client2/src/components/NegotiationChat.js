import React, { useState } from "react";
import { negotiatePrice } from "../services/api";

const NegotiationChat = () => {
  const [message, setMessage] = useState("");
  const [responses, setResponses] = useState([]);

  const handleSend = async () => {
    if (!message.trim()) return;
    const response = await negotiatePrice(message);
    setResponses([...responses, { user: message, bot: response }]);
    setMessage("");
  };

  return (
    <div className="chat-box">
      <h2>💬 Negotiate Your Price</h2>
      <div>
        {responses.map((res, index) => (
          <div key={index}>
            <p><b>You:</b> {res.user}</p>
            <p><b>AI:</b> {res.bot}</p>
          </div>
        ))}
      </div>
      <input 
        value={message} 
        onChange={(e) => setMessage(e.target.value)}
        className="input"
        placeholder="Ask for a discount..."
      />
      <button onClick={handleSend} className="button">Send</button>
    </div>
  );
};

export default NegotiationChat;
