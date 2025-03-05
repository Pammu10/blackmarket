// backend/server.js
const express = require('express');
const cors = require('cors');
const session = require('express-session');
const passport = require('passport');
require('./auth'); // Google OAuth setup
const authRoutes = require('./routes/authRoutes');
const negotiationRoutes = require('./routes/negotiationRoutes');

const app = express();
app.use(express.json());
app.use(cors({ origin: 'http://localhost:3000', credentials: true }));
app.use(session({ secret: 'secretKey', resave: false, saveUninitialized: true }));
app.use(passport.initialize());
app.use(passport.session());

// Routes
app.use('/auth', authRoutes);
app.use('/negotiate', negotiationRoutes);

const PORT = 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
