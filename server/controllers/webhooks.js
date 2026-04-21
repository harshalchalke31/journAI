import Stripe from "stripe";
import Transaction from "../models/Transaction.js";
import User from "../models/User.js";

export const stripeWebhooks = async (req, res) => {
    //initialize stripe
    const stripe = new Stripe(process.env.STRIPE_SECRET_KEY)
    const sign = req.headers["stripe-signature"]
    let event;
    try{
        event = stripe.webhooks.constructEvent(req.body, sign, process.env.STRIPE_WEBHOOK_SECRET)

    }catch(error){
        return res.status(400).send(`Webhook error: ${error.message}`)
    }

    try{
        switch (event.type) {
            case "payment_intent.succeeded":{
                const paymentIntent = event.data.object
                const sessionList = await stripe.checkout.sessions.list({
                    payment_intent: paymentIntent.id,
                })
                const session = sessionList.data[0]
                const {transactioId, appId} = session.metadata
                if(appId==="ragbot"){
                    const transaction = await Transaction.findOne({_id: transactioId, isPaid:false})
                    // update credits in user account
                    await User.updateOne({_id:transaction.userId}, {$inc:{credits: transaction.credits}})
                    // update payment status
                    transaction.isPaid = true
                    await transaction.save()
                }else{
                    return res.json({success: true, message: "Ignored event: Invalid App!"})
                }
            }
            break;
        
            default:
                console.log("Unhandled event type",event.type)
                break;
        }
        res.json({received:true})
    }catch(error){
        console.error("Webhook processing error:", error)
        return res.status(500).send("Internal server error")
    }

}