import express from 'express'
import 'dotenv/config' // to load env variables
import cors from 'cors' //middleware
import connectDB from './configs/db.js'
import userRouter from './routes/userRoutes.js'
import chatRouter from './routes/chatRoutes.js'
import messageRouter from './routes/messageRoutes.js'

//instance
const app = express()

// connect to mongoDB
await connectDB()

//Middleware
app.use(cors())
app.use(express.json()) // all the requests will be passed using json methods

//Routes
app.get('/', (req,res)=> res.send('Server is Live!'))
app.use('/api/user', userRouter)
app.use("/api/chat", chatRouter)
app.use('/api/message', messageRouter)

// Add a port where we will host backend server
const PORT = process.env.PORT || 3000

// Start the express server
app.listen(PORT, ()=>{
    console.log(`Server is running on port ${PORT}`)
})