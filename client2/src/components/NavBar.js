import React, { useContext } from "react";
import { AuthContext } from "../context/AuthContext";
import { Link } from "react-router-dom";

const Navbar = () => {
  const { user, logout } = useContext(AuthContext);

  return (
    <nav className="navbar">
      <Link to="/">🛒 AI eCommerce</Link>
      {user ? (
        <div>
          <span>Hello, {user.name}</span>
          <button onClick={logout} className="button">Logout</button>
        </div>
      ) : (
        <Link to="/login" className="button">Login</Link>
      )}
    </nav>
  );
};

export default Navbar;
