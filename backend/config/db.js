const mysql = require("mysql2");
const dotenv = require("dotenv");

dotenv.config();

const pool = mysql.createPool({
  host: "localhost",
  user: "root",
  password: "password",
  database: "my_schema",
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
});

pool.getConnection((err, connection) => {
  if (err) {
    console.error("Database connection failed:", err);
  } else {
    console.log("✅ MySQL Connected...");
    connection.release();
  }
});

module.exports = pool.promise();
