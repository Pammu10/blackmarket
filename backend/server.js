const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const Sentiment = require("sentiment");
const productRoutes = require('./routes/productRoutes');
const authRoutes = require('./routes/authRoutes');
const path = require('path');
const app = express();
app.use(cors());
app.use(bodyParser.json());

const sentiment = new Sentiment();
let highestOfferedDiscount = 0;
// Sample product discount limits
const productDiscounts = {
  "phone": 30,
  "laptop": 20,
  "headphones": 15
};

neutral_pictures = ["http://localhost:5000/images/neut1.jpeg", "http://localhost:5000/images/neut2.jpeg", "http://localhost:5000/images/neut3.jpg", "http://localhost:5000/images/neut4.jpg", "http://localhost:5000/images/neut5.jpg", "http://localhost:5000/images/neut6.jpg", "http://localhost:5000/images/neut7.webp" ,"http://localhost:5000/images/neut8.gif", "http://localhost:5000/images/neut9.gif", "http://localhost:5000/images/neut10.gif", "http://localhost:5000/images/neut11.jpeg"]
positive_pictures = ['http://localhost:5000/images/yes1.jpeg', 'http://localhost:5000/images/yes2.jpg', 'http://localhost:5000/images/yes3.jpeg', 'http://localhost:5000/images/yes4.jpeg', 'http://localhost:5000/images/yes5.jpeg', 'http://localhost:5000/images/yes6.jpg', 'http://localhost:5000/images/yes7.jpg', 'http://localhost:5000/images/yes8.jpeg', 'http://localhost:5000/images/yes9.jpeg']
negative_pictures = ['http://localhost:5000/images/no1.jpg', 'http://localhost:5000/images/no2.webp', 'http://localhost:5000/images/no3.jpeg', 'http://localhost:5000/images/no4.jpg', 'http://localhost:5000/images/no5.jpg', 'http://localhost:5000/images/no6.jpg', 'http://localhost:5000/images/no7.gif', 'http://localhost:5000/images/no8.png', 'http://localhost:5000/images/no9.jpg', 'http://localhost:5000/images/no10.jpg']

// Reaction image links
const reactionImages = {
  "outrageous": ["http://localhost:5000/images/neut1.jpeg", "http://localhost:5000/images/neut2.jpeg", "http://localhost:5000/images/neut3.jpg", "http://localhost:5000/images/neut4.jpg", "http://localhost:5000/images/neut5.jpg", "http://localhost:5000/images/neut6.jpg", "http://localhost:5000/images/neut7.webp" ,"http://localhost:5000/images/neut8.gif", "http://localhost:5000/images/neut9.gif", "http://localhost:5000/images/neut10.gif", "http://localhost:5000/images/neut11.jpeg"],
  "positive": ['http://localhost:5000/images/yes1.jpeg', 'http://localhost:5000/images/yes2.jpg', 'http://localhost:5000/images/yes3.jpeg', 'http://localhost:5000/images/yes4.jpeg', 'http://localhost:5000/images/yes5.jpeg', 'http://localhost:5000/images/yes6.jpg', 'http://localhost:5000/images/yes7.jpg', 'http://localhost:5000/images/yes8.jpeg', 'http://localhost:5000/images/yes9.jpeg'],
  "negative": ['http://localhost:5000/images/no1.jpg', 'http://localhost:5000/images/no2.webp', 'http://localhost:5000/images/no3.jpeg', 'http://localhost:5000/images/no4.jpg', 'http://localhost:5000/images/no5.jpg', 'http://localhost:5000/images/no6.jpg', 'http://localhost:5000/images/no7.gif', 'http://localhost:5000/images/no8.png', 'http://localhost:5000/images/no9.jpg', 'http://localhost:5000/images/no10.jpg'],
  "neutral": ["http://localhost:5000/images/neut1.jpeg", "http://localhost:5000/images/neut2.jpeg", "http://localhost:5000/images/neut3.jpg", "http://localhost:5000/images/neut4.jpg", "http://localhost:5000/images/neut5.jpg", "http://localhost:5000/images/neut6.jpg", "http://localhost:5000/images/neut7.webp" ,"http://localhost:5000/images/neut8.gif", "http://localhost:5000/images/neut9.gif", "http://localhost:5000/images/neut10.gif", "http://localhost:5000/images/neut11.jpeg"]
};

function getRandomReaction(mood) {
    const images = reactionImages[mood] || reactionImages["neutral"];
    return images[Math.floor(Math.random() * images.length)];
  }

// Negotiation function
function negotiate(userInput, requestedDiscount, product) {
    console.log(highestOfferedDiscount);
    const maxDiscount = productDiscounts[product] || 40;
    let sentimentResult = sentiment.analyze(userInput);
    let sentimentScore = sentimentResult.score;
  
    // Simple keyword-based detection for greetings or general talk
    const greetings = ["hello", "hey", "good morning", "good evening"];
    const generalTalk = ["how are you", "what’s up", "how’s it going", "tell me a joke"];
    const higherKeyWords = ["higher", "more", "increase", "raise", "up", "bigger", "greater", "enhance", "boost", "expand", "augment", "intensify", "amplify", "heighten", "escalate", "inflate", "elevate", "augment", "intensify", "amplify", "heighten", "escalate", "inflate", "elevate"];
    const discountTalk = [ "discount", "offer", "deal", "price", "cost", "rate", "cut", "reduce", "lower", "decrease", "drop", "fall", "less", "cheap", "affordable", "budget", "save", "economize", "discounted", "sale", "special", "promotional", "cheaper", "affordability", "afford", "affordable", "budget-friendly", "budget-conscious", "budget-conscious"]
    let lowerInput = userInput.toLowerCase();
    let mood = "neutral";
    if (sentimentScore > 0) mood = "positive";
    if (sentimentScore < 0) mood = "negative";
    let reaction = getRandomReaction(mood);
    console.log(reaction);
    // Handle greetings
    if (greetings.some((greet) => lowerInput.includes(greet))) {
      return { message: "Hello there! Ready to strike a deal? 😉", reaction };
    }
    
    
    // Handle joke request
    if (lowerInput.includes("joke")) {
      return { 
        message: "Why did the bargaining AI refuse to give a big discount? Because it didn't want to go 'too low'! 😆", 
        reaction 
      };
    }

    if (discountTalk.some((talk) => lowerInput.includes(talk))) {
        // If the request is about an unrealistic discount
    if (requestedDiscount > maxDiscount) {
        reaction = reactionImages["outrageous"];
        return { message: "That's too much! I can't go that low. 🤯", reaction };
      }
      // Regular negotiation response
      let counterOffer = Math.max(requestedDiscount - 5, 5);
      if (counterOffer < highestOfferedDiscount) {
        counterOffer = highestOfferedDiscount;
        return {
            message: `I can offer you the previously offered ${counterOffer}% discount instead! 😉`,
            reaction
          };
      }
      highestOfferedDiscount = Math.max(counterOffer, highestOfferedDiscount);
      return {
        message: `I can offer you ${counterOffer}% discount instead! 😉`,
        reaction
      };
      
    }
    if (higherKeyWords.some((word) => lowerInput.includes(word))) {
       
        reaction = reactionImages["negative"]; 
        let increasedDiscount = Math.min(requestedDiscount + 5, maxDiscount);
        let decreasedDiscount = Math.max(requestedDiscount - 5, 5);
        const responses = [
            `I can go higher than that. Is a discount of ${increasedDiscount}% acceptable?`,
           `I can't go higher than that. 😞 The discount I am okay with is ${decreasedDiscount}%.`
          ];
        const message = responses[Math.floor(Math.random() *2)];
        if (increasedDiscount < highestOfferedDiscount || decreasedDiscount < highestOfferedDiscount) return { message: "I can't go higher than that. 😞 The discount I am okay with is " + highestOfferedDiscount + "%.", reaction };
        highestOfferedDiscount = Math.max(highestOfferedDiscount, message.discount);
        return { message, reaction };
      }
      
      
    
    return { message: "Let's talk business shall we? 🤝", reaction };
    
}
  
app.use('/auth', authRoutes);
app.use('/products', productRoutes)
app.use('/images', express.static(path.join(__dirname, '/public/reactions')));

// API endpoint
app.post("/bargain", (req, res) => {
  const { userInput, requestedDiscount, product } = req.body;
  const response = negotiate(userInput, requestedDiscount, product);
  res.json(response);
});

app.listen(5000, () => console.log("Server running on port 5000"));


// const express = require('express');
// const cors = require('cors');
// const session = require('express-session');
// const passport = require('passport');
// require('./auth'); // Google OAuth setup
// const authRoutes = require('./routes/authRoutes');
// const negotiationRoutes = require('./routes/negotiationRoutes');
// const qlearnroutes = require('./routes/rlNegotiations');
// const productRoutes = require('./routes/productRoutes');


// const app = express();
// app.use(express.json());
// app.use(cors());
// app.use(session({ secret: 'secretKey', resave: false, saveUninitialized: true }));
// app.use(passport.initialize());
// app.use(passport.session());

// // Routes
// app.use('/auth', authRoutes);
// app.use('/negotiate', negotiationRoutes);
// app.use('/qlearn', qlearnroutes);
// app.use('/products', productRoutes)

// const PORT = 5000;
// app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
