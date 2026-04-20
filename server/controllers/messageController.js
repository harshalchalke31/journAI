import Chat from "../models/Chat.js"
import llm1 from '../configs/aiModels.js'
import User from "../models/User.js"
import axios from 'axios'
import imagekit from "../configs/imagekit.js"

//text based ai chat message controller
export const textMessageController = async (req, res) => {
    try {
        //fetch userId, chatId and prompt for inference
        const userId = req.user._id
        //check credits
        if (req.user.credits < 1) {
            return res.json({ success: false, message: "You don't have enough credits!" })
        }
        const { chatId, prompt } = req.body

        const chat = await Chat.findOne({ userId, _id: chatId })
        chat.messages.push({
            role: "user", content: prompt, timestamp: Date.now(),
            isImage: false
        })

        const response = await llm1.models.generateContent({
            model: "gemini-3-flash-preview",
            contents: prompt,
        })

        const reply = { role:"assistant", content: response.text, timestamp: Date.now(), isImage: false }
        res.json({ success: true, reply })

        chat.messages.push(reply)
        await chat.save()
        await User.updateOne({ _id: userId }, { $inc: { credits: -1 } })
    } catch (error) {
        res.json({ success: false, message: error.message })
    }
}

export const imageMessageController = async (req, res) => {
    try {
        const userId = req.body._id
        //check credits
        if (req.user.credits < 2) {
            return res.json({ success: false, message: "You don't have enough credits!" })
        }
        const { prompt, chatId, isPublished } = req.body
        //find chat
        const chat = await Chat.findOne({ userId, _id: chatId })
        //push chat
        chat.messages.push({
            role: "user",
            content: prompt,
            timestamp: Date.now(),
            isImage: true
        })
        //encode the prompt
        const encodedPrompt = encodeURIComponent(prompt)
        //construct Image generation URL for ImageKit
        const generatedImageUrl = `${process.env.IMAGEKIT_URL_ENDPOINT}/ik-genimg-prompt-${encodedPrompt}/
        ragbot/${Date.now()}.png?tr=w-800,h-800`

        //trigger generation by fetching imagekit
        const aiImageResponse = await axios.get(generatedImageUrl, {responseType:'arraybuffer'})

        //convert to base64 image for upload
        const base64Image = `data:image/png;base64,${Buffer.from(aiImageResponse.data, 'binary').toString('base64')}`

        //upload to imagekit library
        const uploadResponse = await imagekit.upload({
            file:base64Image,
            fileName: `${Date.now()}.png`,
            folder: 'ragbot'
        })

        const reply = {
            role: 'assistant',
            content: uploadResponse.url,
            timestamp: Date.now(),
            isImage:true,
            isPublished
        }

        res.json({success:true, reply})
        chat.messages.push(reply)
        await chat.save()
        await User.updateOne({_id:userId},{$inc:{credits:-2}})

    } catch (error) {
        res.json({success:false, message:error.message})
    }
}