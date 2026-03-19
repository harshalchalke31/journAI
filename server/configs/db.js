import mongoose from 'mongoose'

// create an async function to connect and query to MongoDB
const connectDB = async()=>{
    try{
        //adding an event to test connection
        mongoose.connection.on('connected', ()=>console.log('MongoDB connection successful!'))
        await mongoose.connect(`${process.env.MONGODB_URI}/ragbot`)
    }catch(error){
        console.log(error.message)
    }
}

export default connectDB