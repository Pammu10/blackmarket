import React, { createContext, useState, useEffect } from "react";
import { loginWithGoogle, logoutUser, fetchUser } from "../services/api";

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);

  useEffect(() => {
    fetchUser().then(setUser);
  }, []);

  return (
    <AuthContext.Provider value={{ user, login: loginWithGoogle, logout: logoutUser }}>
      {children}
    </AuthContext.Provider>
  );
};
