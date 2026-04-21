import express from 'express'
import 'dotenv/config' // to load env variables
import cors from 'cors' //middleware
import connectDB from './configs/db.js'
import userRouter from './routes/userRoutes.js'
import chatRouter from './routes/chatRoutes.js'
import messageRouter from './routes/messageRoutes.js'
import creditRouter from './routes/creditRoutes.js'
import { stripeWebhooks } from './controllers/webhooks.js'

//instance
const app = express()

// connect to mongoDB
await connectDB()

//Stripe webhooks
app.post('/api/stripe',express.raw({type:'application/json'}), stripeWebhooks)

//Middleware
app.use(cors())
app.use(express.json()) // all the requests will be passed using json methods

//Routes
app.get('/', (req,res)=> res.send('Server is Live!'))
app.use('/api/user', userRouter)
app.use("/api/chat", chatRouter)
app.use('/api/message', messageRouter)
app.use('/api/credit', creditRouter)

// Add a port where we will host backend server
const PORT = process.env.PORT || 3000

// Start the express server
app.listen(PORT, ()=>{
    console.log(`Server is running on port ${PORT}`)
})