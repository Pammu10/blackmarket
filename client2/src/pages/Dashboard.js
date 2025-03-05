import React, { useContext } from "react";
import { AuthContext } from "../context/AuthContext";

const Dashboard = () => {
  const { user } = useContext(AuthContext);

  return (
    <div className="dashboard">
      <h1>📊 Dashboard</h1>
      <p>
        {user.role === "seller" 
          ? "Access your seller panel to set negotiation rules."
          : "Enjoy exclusive AI-powered deals!"}
      </p>
    </div>
  );
};

export default Dashboard;
