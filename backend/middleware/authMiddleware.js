// backend/middleware/authMiddleware.js
const jwt = require('jsonwebtoken');

// Middleware to authenticate user via JWT
function authenticateUser(req, res, next) {
    const token = req.header('Authorization');
    if (!token) return res.status(401).json({ message: 'Access denied' });

    try {
        const verified = jwt.verify(token.replace('Bearer ', ''), 'jwtSecret');
        req.user = verified;
        next();
    } catch (err) {
        res.status(400).json({ message: 'Invalid token' });
    }
}

// Middleware to authorize specific roles
function authorizeRole(role) {
    return (req, res, next) => {
        if (req.user.role !== role) {
            return res.status(403).json({ message: 'Forbidden' });
        }
        next();
    };
}

module.exports = { authenticateUser, authorizeRole };
