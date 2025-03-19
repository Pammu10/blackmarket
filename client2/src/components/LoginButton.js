import React, { useContext } from "react";
import { AuthContext } from "../context/AuthContext";

const LoginButton = () => {
  const { login } = useContext(AuthContext);

  return (
    <button onClick={login} className="bg-green-500 px-4 py-2 rounded">
      Login with Google
    </button>
  );
};

export default LoginButton;
