// backend/server.js
const express = require('express');
const cors = require('cors');
const session = require('express-session');
const passport = require('passport');
require('./auth'); // Google OAuth setup
const authRoutes = require('./routes/authRoutes');
const negotiationRoutes = require('./routes/negotiationRoutes');
const qlearnroutes = require('./routes/rlNegotiations');
const productRoutes = require('./routes/productRoutes');


const app = express();
app.use(express.json());
app.use(cors());
app.use(session({ secret: 'secretKey', resave: false, saveUninitialized: true }));
app.use(passport.initialize());
app.use(passport.session());

// Routes
app.use('/auth', authRoutes);
app.use('/negotiate', negotiationRoutes);
app.use('/qlearn', qlearnroutes);
app.use('/products', productRoutes)

const PORT = 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
