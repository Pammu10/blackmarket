const express = require("express");
const db = require("../config/db");

const router = express.Router();

// Get All Products
router.get("/", async (req, res) => {
  try {
    const [products] = await db.execute("SELECT * FROM products");
    res.json(products);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch products" });
  }
});

module.exports = router;
