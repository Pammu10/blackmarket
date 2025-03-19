import React, { createContext, useState, useEffect } from "react";

export const AuthContext = createContext();


export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);

  const storeToken = (token) => {
      localStorage.setItem("token", token);
    }

  const storeUser = (userData) => {
    const {id, name, role } = userData;
    setUser({id, name, role});

  }

  const signIn = (data) => {
    storeToken(data.token);
    storeUser(data.user);
  }

  



  useEffect(() => {
    const token = localStorage.getItem("token");
    if (token) {
      setUser(token);
    }
  }, []);
  
  useEffect(() => {
    console.log("USER CHANGES")
    console.log(user);
  }, [user]); 
  const deleteToken = () => {
    localStorage.removeItem("token");
  }
  const logoutUser = () => {
    setUser(null);
    deleteToken();
  }


  return (
    <AuthContext.Provider value={{ user, signIn: signIn, logout: logoutUser }}>
      {children}
    </AuthContext.Provider>
  );
};
