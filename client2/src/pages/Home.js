import React from "react";
import Navbar from "../components/NavBar"
import NegotiationChat from "../components/NegotiationChat";

const Home = () => {
  return (
    <div className="bg-gray-100 min-h-screen">
      <Navbar />
      <div className="container mx-auto text-center py-12">
        <h1 className="text-4xl font-bold text-gray-800">Welcome to AI-Powered eCommerce</h1>
        <p className="text-gray-600 mt-2">Negotiate prices in real time with AI.</p>
      </div>
      <NegotiationChat />
    </div>
  );
};

export default Home;
