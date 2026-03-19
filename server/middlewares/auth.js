import jwt from 'jsonwebtoken'
import User from '../models/User.js'

// this is a middleware called before route to protect the user data
// this way only authorized users can have access to the route

export const protect = async (req, res, next) => {
    let token = req.headers.authorization
    try{
        const decoded = jwt.verify(token, process.env.JWT_SECRET)
        const userId = decoded.id
        const user = await User.findById(userId)

        if(!user){
            return res.json({success:false, message:'Not authorized! User not found.'})
        }
        req.user = user
        // next()

    }catch(error){
        res.status(401).json({message:'Not authorized, token failed.'})
    }
}