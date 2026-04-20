import Chat from "../models/Chat.js"

//API controller for creating new chat
export const createChat = async (req, res) => {
    try {
        //fetch user id
        const userId = req.user._id

        //initialize chat data
        const chatData = {
            userId,
            messages: [],
            name: 'New Chat',
            userName: req.user.name,
        }

        await Chat.create(chatData)
        res.json({ success: true, message: 'Chat created.' })
    } catch (error) {
        res.json({ success: false, message: error.message })
    }
}

//API controller for getting chats
export const getChats = async (req, res) => {
    try {
        //get userid for reference
        const userId = req.user._id
        //find the requested chat
        const chats = await Chat.find({ userId }).sort({ updatedAt: -1 })
        //return chats
        res.json({ success: true, chats })
    } catch (error) {
        res.json({ success: false, message: error.message })
    }
}

//API controller for deleting a chat
export const deleteChats = async(req,res)=>{
    try{
        //get userId and chatId for reference
        const userId = req.user._id
        const chatId = req.body
        await Chat.deleteOne({_id:chatId, userId})
        res.json({success:true, message:'Chat deleted.'})
    }catch(error){
        res.json({success:false, message:error.message})
    }
}