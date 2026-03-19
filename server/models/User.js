import mongoose from "mongoose";
import bcrypt from 'bcryptjs'

// user schema for User table
const userSchema = new mongoose.Schema({
    name: {type: String, required: true},
    email: {type: String, required: true, unique: true},
    password: {type: String, required: true},
    credits: {type: Number, default: 200},

})

// hashing password before upload
userSchema.pre('save', async function () {
    if(!this.isModified('password')){
        return //next()
    }
    const salt = await bcrypt.genSalt(10)
    this.password = await bcrypt.hash(this.password, salt)
    //next();
    
})

// add the data model to table
const User = mongoose.model('User', userSchema)

export default User