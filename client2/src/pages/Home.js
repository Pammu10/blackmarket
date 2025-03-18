import React from "react";
import Navbar from "../components/NavBar"
import NegotiationChat from "../components/NegotiationChat";
import BargainApp from "../components/BargainApp";

const Home = () => {
  return (
    <div className="bg-gray-100 min-h-screen">
      <Navbar />
      <div className="chat-box-header">
        <h1 className="centering">Welcome to AI-Powered eCommerce</h1>
        <p className="centering">Negotiate prices in real time with AI.</p>
      </div>
      {/* <NegotiationChat /> */}
      <BargainApp/>
    </div>
  );
};

export default Home;
