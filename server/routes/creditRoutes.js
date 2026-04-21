import express from 'express'
import { getPlans, purschasePlan } from '../controllers/creditController.js'
import {protect} from '../middlewares/auth.js'

const creditRouter = express.Router()

creditRouter.get('/plan',getPlans)
creditRouter.post('/purchase', protect, purschasePlan)

export default creditRouter