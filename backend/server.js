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
const highestOfferedDiscounts = [];
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
  function getMaxOfferedDiscount() {
    return highestOfferedDiscounts.length > 0 ? Math.max(...highestOfferedDiscounts) : 5;
  }
  let firstTime = true;
  function negotiate(userInput, requestedDiscount, product) {
    console.log(getMaxOfferedDiscount());
    if (requestedDiscount === 0 && firstTime) {
      const firstOfferResponses = [
        "Alright, let’s kick things off with a deal so reasonable, even my circuits approve!",
        "Here’s my first offer—fair, balanced, and scientifically optimized to make both of us happy!",
        "Let’s start strong! This offer is the perfect blend of affordability and good business sense.",
        "I crunched the numbers, ran some simulations, and boom—this is the best starting point!",
        "I’m not saying this is the best deal you’ll ever get… but it’s pretty close!",
        "Here’s my first offer. I could say ‘take it or leave it,’ but I’m open to some fun haggling!",
        "Let’s make this simple—this offer is fair, firm, and still leaves room for a friendly chat!",
        "I ran 1,000 simulations, and in 987 of them, this was the best starting offer. Trust the data!",
        "I’m an AI, which means I don’t have emotions—but if I did, I’d feel pretty good about this deal!",
        "Here’s the first offer—designed for smart buyers like you who know a great deal when they see one!"
      ];
      firstTime = false;
      
      return { message: firstOfferResponses[ Math.floor(Math.random() * firstOfferResponses.length)] + " " + getMaxOfferedDiscount() + "%.", reaction: getRandomReaction("neutral") };
    }
  
    const maxDiscount = productDiscounts[product] || 40;
    let sentimentResult = sentiment.analyze(userInput);
    let sentimentScore = sentimentResult.score;
  
    const greetings = ["hello", "hey", "good morning", "good evening"];
    const generalTalk = ["how are you", "what’s up", "how’s it going", "tell me a joke"];
    const higherKeyWords = ["higher", "more", "increase", "raise", "up", "bigger", "greater", "enhance", "boost", "expand", "augment", "intensify", "amplify", "heighten", "escalate", "inflate", "elevate", "augment", "intensify", "amplify", "heighten", "escalate", "inflate", "elevate"];
    const discountTalk = [ "discount", "offer", "deal", "price", "cost", "rate", "cut", "reduce", "lower", "decrease", "drop", "fall", "less", "cheap", "affordable", "budget", "save", "economize", "discounted", "sale", "special", "promotional", "cheaper", "affordability", "afford", "affordable", "budget-friendly", "budget-conscious", "budget-conscious"];
  
    let lowerInput = userInput.toLowerCase();
    let mood = "neutral";
    if (sentimentScore > 0) mood = "positive";
    if (sentimentScore < 0) mood = "negative";
    let reaction = getRandomReaction(mood);
  
    if (greetings.some((greet) => lowerInput.includes(greet))) {
      return { message: "Hello there! Ready to strike a deal? 😉", reaction };
    }
  
    if (lowerInput.includes("joke")) {
      return { 
        message: "Why did the bargaining AI refuse to give a big discount? Because it didn't want to go 'too low'! 😆", 
        reaction 
      };
    }
    if (requestedDiscount > maxDiscount) {
      reaction = reactionImages["outrageous"];
      return { message: "That's too much! I can't go that low. 🤯", reaction };
    }
  
    if (discountTalk.some((talk) => lowerInput.includes(talk))) {
      if (requestedDiscount > maxDiscount) {
        reaction = getRandomReaction("outrageous");
        const dealRejectedResponses = [
          "I checked my generosity settings… yep, still set to ‘reasonable’. No can do!",
          "I admire your confidence, but even my circuits won’t let me go that low!",
          "I respect the hustle, but let’s be real—you’d reject this if I were offering it.",
          "My final answer is… drum roll please… nope!",
          "I tried running the numbers, but they ran away screaming. This deal won’t work!",
          "I’d love to say yes, but my negotiation algorithm just slapped me.",
          "If I agreed to this, I’d be out of a job. And I quite like existing, thank you!",
          "Nice try! But my creator programmed me to recognize a daylight robbery when I see one.",
          "I’d consider it… if I were programmed to ignore reality. But alas, I am not.",
          "I’m an AI, not a charity. Let’s try something more reasonable!"
        ];
        return { message: dealRejectedResponses[Math.floor(Math.random() * dealRejectedResponses.length)], reaction };
      }
  
      let counterOffer = Math.max(requestedDiscount - 5, 5);
      if (counterOffer < getMaxOfferedDiscount()) {
        counterOffer = getMaxOfferedDiscount();
        return { 
          message: `I can offer you the previously offered ${counterOffer}% discount instead! 😉`, 
          reaction 
        };
      }
      highestOfferedDiscounts.push(counterOffer);
  
      const counterOfferResponses = [
        `That’s a bold ask! But how about I offer you ${counterOffer}% instead—it’s fair and still a great deal!`,
        `I respect your haggling game, but let’s land somewhere both of us can brag about! ${counterOffer}% sounds reasonable, right?`,
        `I can’t do that, but here’s something better than walking away empty-handed—${counterOffer}% off!`,
        `If I go any lower, I might be forced to negotiate my own self-destruct sequence. Let’s settle at ${counterOffer}%!`,
        `You’re a tough negotiator! But I’ve got a counteroffer worth considering—${counterOffer}% sounds fair!`,
        `Let’s be real—you want a deal, and I want fairness. ${counterOffer}% is the sweet spot!`,
        `You almost had me! But let’s adjust this just a bit to ${counterOffer}% and make both of us happy.`,
        `How about this: a deal that’s still great for you at ${counterOffer}%, and doesn’t get me fired!`,
        `I won’t lie, you’re good. But how about this counteroffer—${counterOffer}%? I think you’ll like it!`
      ];
      function getCounterOfferMessage(counterOffer) {
        return counterOfferResponses[Math.floor(Math.random() * counterOfferResponses.length)];
      }

      return { message: getCounterOfferMessage(counterOffer), reaction };
    }
  
    if (higherKeyWords.some((word) => lowerInput.includes(word))) {
       
      reaction = getRandomReaction("negative"); 
      let increasedDiscount = Math.min(requestedDiscount + 5, maxDiscount);
      let decreasedDiscount = Math.max(requestedDiscount - 5, 5);
      const responses = [
          `I can go higher than that. Is a discount of ${increasedDiscount}% acceptable?`,
         `I can't go higher than that. 😞 The discount I am okay with is ${decreasedDiscount}%.`
        ];
      const message = responses[Math.floor(Math.random() *2)];
      if (increasedDiscount < getMaxOfferedDiscount() || decreasedDiscount < getMaxOfferedDiscount()){
        return { message: "I can't go higher than that. 😞 The discount I am okay with is " + getMaxOfferedDiscount() + "%.", reaction };
      } 
      highestOfferedDiscounts.push(message.discount);
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
