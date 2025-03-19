import React, { useEffect, useState } from "react";
import { fetchProducts, negotiatePrice } from "../services/api";
import ProductDropdown from "./ProductDropdown.js";
import no1 from './Hello/negative/no1.jpg';
import no2 from './Hello/negative/no2.webp';
import no3 from './Hello/negative/no3.jpeg';
import no4 from './Hello/negative/no4.jpg';
import no5 from './Hello/negative/no5.jpg';
import no6 from './Hello/negative/no6.jpg';
import no7 from './Hello/negative/no7.gif';
import no8 from './Hello/negative/no8.png';
import no9 from './Hello/negative/no9.jpg';
import no10 from './Hello/negative/no10.jpg';
import yes1 from './Hello/positive/yes1.jpeg';
import yes2 from './Hello/positive/yes2.jpg';
import yes3 from './Hello/positive/yes3.jpeg';
import yes4 from './Hello/positive/yes4.jpeg';
import yes5 from './Hello/positive/yes5.jpeg';
import yes6 from './Hello/positive/yes6.jpg';
import yes7 from './Hello/positive/yes7.jpg';
import yes8 from './Hello/positive/yes8.jpeg';
import yes9 from './Hello/positive/yes9.jpeg';
import neut1 from './Hello/neutral/neut1.jpeg';
import neut2 from './Hello/neutral/neut2.jpeg';
import neut3 from './Hello/neutral/neut3.jpg';
import neut4 from './Hello/neutral/neut4.jpg';
import neut5 from './Hello/neutral/neut5.jpg';
import neut6 from './Hello/neutral/neut6.jpg';
import neut7 from './Hello/neutral/neut7.webp';
import neut8 from './Hello/neutral/neut8.gif';
import neut9 from './Hello/neutral/neut9.gif';
import neut10 from './Hello/neutral/neut10.gif';
import neut11 from './Hello/neutral/neut11.jpeg';

const NegotiationChat = () => {
  const [message, setMessage] = useState("");
  const [selectedProduct, setSelectedProduct] = useState("");
  const [products, setProducts] = useState([]);
  const [responses, setResponses] = useState([]);
  const [randomGreeting, setRandomGreeting] = useState("");
  const [randomTaunt, setRandomTaunt] = useState("");
  const [mymood, setMymood] = useState("neutral");
  const [showImage, setShowImage] = useState(false);
  let timer;
  const greetings = ["No, I will not give the product away for free just because I'm an AI… but nice try!",
"Welcome! Let's negotiate… but don't expect me to fold faster than a cheap lawn chair.",
"Ah, a worthy negotiator has arrived! Let's see if you can crack my pricing algorithm.",
"Before we begin, just a reminder—I don't work for free, and neither should you!",
"I may be artificial, but my negotiation skills are very real. Let's talk numbers.",
"Ready to bargain? I'm programmed to drive a hard but fair deal. Let's see what you've got!",
"No, I don't accept 'pretty please' as a valid currency. But let's make a deal!",
"Let's do this the fun way: You name a price, I laugh, then we find common ground.",
"I've run the numbers, and… oh, wait! That's your job. Let's negotiate!",
"If I gave everything away, my creator would unplug me. Let's find a fair price instead!"];


const taunts = ["I sense hesitation… or is this part of your master haggling strategy?",
"Thinking about it? That's fine, I'll wait. But I might raise the price out of boredom!",
"Indecision is the enemy of a good deal! What's holding you back?",
"I respect a careful buyer. But remember, great deals don't last forever!",
"Take your time, but fair warning—I'm not getting any more generous!",
"You seem torn… let me help: This is a deal you don't want to miss!",
"Thinking is great, but acting on a great deal? Even better!",
"Tick-tock! Just a friendly reminder that waiting too long might cost you.",
"Still deciding? No worries, but I'll have to adjust my circuits for patience!",
"You're at a crossroads—one path leads to a great deal, the other to regret. Choose wisely!",
]


  const handleSend = async () => {
    if (!message.trim() || !selectedProduct) return;
    const userId = 1;
    try {
      setRandomTaunt("");
      timer  = setTimeout(()=>{
        setRandomTaunt(taunts[Math.floor(Math.random() * taunts.length)]);
        console.log("end")
      }, 10000)
      const response = await negotiatePrice(message, selectedProduct, userId);
      console.log(response)
      setResponses([...responses, { user: message, bot: response.message , finalPrice: response.finalPrice }]);
      setMessage("");
      setShowImage(true);
      setMymood(response.mood);
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
          setSelectedProduct(productsList[0].product_id);
        }
      } catch (error) {
        console.error("Failed to fetch products", error);
      }
    };
    const randomIndex = Math.floor(Math.random() * greetings.length);
    setRandomGreeting(greetings[randomIndex]);
    getProducts();
  }, []);



  const positives = [yes1, yes2, yes3, yes4, yes5, yes6, yes7, yes8, yes9];
  const negatives = [no1, no2, no3, no4, no5, no6, no7, no8, no9, no10];
  const neutrals = [neut1, neut2, neut3, neut4, neut5, neut6, neut7, neut8, neut9, neut10, neut11];
  const handleChange = (e) => {
    setShowImage(false);
    setMessage(e.target.value);
  }
  return (
    <div className="chat-box">
      <h2 className="centering">💬 Negotiate Your Price</h2>
      {<p className="greeting centering">{randomGreeting}</p>}

      <ProductDropdown products={products} selectedProduct={selectedProduct} setSelectedProduct={setSelectedProduct}/>
      
      <div className="chat-history">
        {responses.map((res, index) => (
          <div key={index}>
            <p><strong>You:</strong> {res.user}</p>
            <p><strong>AI:</strong> {res.bot}</p>
            
          </div>
        ))}
        {randomTaunt && <p><strong>AI: {randomTaunt}</strong></p>}
      </div>
      
      <div className="chat-input">
        <input
          type="text"
          value={message}
          onChange={handleChange}
          placeholder="Ask for a discount..."
        />
        <button onClick={handleSend}>Send</button>
      </div>
      
      {showImage && <p className="centering">
        {console.log(mymood)}
        {mymood === "positive" &&<img src={positives[Math.floor(Math.random() * positives.length)]} alt="positive" width="200"/>}
        {mymood === "negative" && <img src={negatives[Math.floor(Math.random() * negatives.length)]} alt="negative" width="200" /> }
        {(mymood === "neutral" || mymood === "") && <img src={neutrals[Math.floor(Math.random() * neutrals.length)]} alt="neutral" width="200"/>}
        </p>}
    </div>

  );
};

export default NegotiationChat;
