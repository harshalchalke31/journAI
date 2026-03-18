import express from 'express'
import 'dotenv/config' // to load env variables
import cors from 'cors' //middleware

//instance
const app = express()

//Middleware
app.use(cors())
app.use(express.json()) // all the requests will be passed using json methods

//Routes
app.get('/', (req,res)=> res.send('Server is Live!'))

// Add a port where we will host backend server
const PORT = process.env.PORT || 3000

// Start the express server
app.listen(PORT, ()=>{
    console.log(`Server is running on port ${PORT}`)
})