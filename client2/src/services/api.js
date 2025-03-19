import axios from "axios";

const API_URL = "http://localhost:5000";


export const login = async (email, password) => {
  console.log("logging in")
  console.log(email, password)
  const res = await axios.post(`${API_URL}/auth/login`, { email, password });
  return res.data;
}


export const register = async (email, password, name) => { 
  console.log("registering")
  console.log(email, password, name)
  const res = await axios.post(`${API_URL}/auth/register`, { name, email, password });
  return res.data;
}


export const loginWithGoogle = async () => {
  window.location.href = `${API_URL}/auth/google`;
};

export const logoutUser = async () => {
  await axios.post(`${API_URL}/auth/logout`);
  window.location.reload();
};

export const fetchUser = async () => {
  const res = await axios.get(`${API_URL}/auth/user`);
  return res.data;
};

export const negotiatePrice = async (message, productId, userId) => {
  const res = await axios.post(`${API_URL}/qlearn`, { message, userId, productId });
  return res.data;
};

export const setNegotiationRule = async (rule) => {
  await axios.post(`${API_URL}/seller/rules`, rule);
};

export const fetchProducts = async () => {
  const res = await axios.get(`${API_URL}/products`);
  console.log(res.data);
  return res.data;
};