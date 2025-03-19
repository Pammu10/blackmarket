const express = require("express");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const db = require("../config/db");

const router = express.Router();

// User Registration
router.post("/register", async (req, res) => {
  const { name, email, password } = req.body;
  const hashedPassword = await bcrypt.hash(password, 10);

  try {
    const [result] = await db.execute(
      "INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)",
      [name, email, hashedPassword, "user"]
    );
    console.log(result);
    const token = jwt.sign({ id: result.insertId, role: "user" }, 'password', { expiresIn: "1h" });
    res.json({ token, user: { id: result.insertId, name: name, role: "user" }});
  } catch (error) {
    console.log(error)
    res.status(500).json({ error: "Registration failed" });
  }
});

// User Login
router.post("/login", async (req, res) => {
  const { email, password } = req.body;
  try {
    const [users] = await db.execute("SELECT * FROM users WHERE email = ?", [email]);
    if (users.length === 0) return res.status(401).json({ error: "User not found" });
    const user = users[0];
    const passwordMatch = await bcrypt.compare(password, user.password);
    if (!passwordMatch) return res.status(401).json({ error: "Invalid credentials" });
    const token = jwt.sign({ id: user.id, role: user.role }, 'password', { expiresIn: "1h" });
    res.json({ token, user: { id: user.id, name: user.name, role: user.role } });
  } catch (error) {
    console.log("CAUGHT", error);
    res.status(500).json({ error: "Login failed" });
  }
});

module.exports = router;
