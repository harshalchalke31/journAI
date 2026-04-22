import { createContext, useContext, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { dummyUserData, dummyChats } from '../assets/assets'
import toast from 'react-hot-toast'
import axios from 'axios';
axios.defaults.baseURL = import.meta.env.VITE_SERVER_URL

const AppContext = createContext()
export const AppContextProvider = ({ children }) => {

    const navigate = useNavigate()
    const [user, SetUser] = useState(null)
    const [chats, SetChats] = useState([])
    const [selectedChat, SetSelectedChat] = useState(null)
    const [theme, SetTheme] = useState(localStorage.getItem('theme') || 'light')

    //API call to server to fetch user
    const [token, setToken] = useState(localStorage.getItem("token") || null) //get token if available
    const [loadingUser, setLoadinguser] = useState(true)
    const fetchUser = async () => {
        try {
            const { data } = await axios.get('/api/user/data', { headers: { Authorization: token } })
            if (data.success) {
                SetUser(data.user)
            } else {
                toast.error(data.message)
            }
        } catch (error) {
            toast.error(error.message)
        } finally {
            setLoadinguser(false)
        }
    }

    // //API cal to create a new chat
    // const createNewChat = async () => {
    //     try {
    //         if (!user) return toast.error("Login to create a new chat!")
    //         navigate('/')
    //         await axios.get('/api/chat/create', { headers: { Authorization: token } })
    //         await fetchUsersChats()
    //     } catch (error) {
    //         toast.error(error.message)
    //     }
    // }

    // //API call to server to fetch user chats
    // const fetchUsersChats = async () => {
    //     try {
    //         const data = await axios.get('/api/chat/get', { headers: { Authorization: token } })
    //         if (data.success) {
    //             //if user has no chats, create one
    //             if (data.chats.length === 0) {
    //                 await createNewChat()
    //             } else {
    //                 SetChats(data.chats)
    //                 SetSelectedChat(data.chats[0])
    //             }
    //         } else {
    //             toast.error("No data")
    //         }
    //     } catch (error) {
    //         toast.error(error.message)
    //     }
    // }

    // const createNewChat = async () => {
    //     try {
    //         if (!user) return toast.error("Login to create a new chat!")
    //         await axios.get('/api/chat/create', { headers: { Authorization: token } })
    //         // ❌ Remove fetchUsersChats() from here
    //     } catch (error) {
    //         toast.error(error.message)
    //     }
    // }

    // const fetchUsersChats = async () => {
    //     try {
    //         const { data } = await axios.get('/api/chat/get', { headers: { Authorization: token } })

    //         if (data.success) {
    //             if (data.chats.length === 0) {
    //                 await createNewChat()       // create the chat
    //                 // Now fetch again ONCE, no loop because
    //                 // createNewChat no longer calls fetchUsersChats
    //                 const { data: newData } = await axios.get('/api/chat/get', { headers: { Authorization: token } })
    //                 SetChats(newData.chats)
    //                 SetSelectedChat(newData.chats[0])
    //             } else {
    //                 SetChats(data.chats)
    //                 SetSelectedChat(data.chats[0])
    //             }
    //         } else {
    //             toast.error("Failed to fetch chats")
    //         }
    //     } catch (error) {
    //         toast.error(error.message)
    //     }
    // }
    useEffect(() => {
        if (token) {
            fetchUser()
        } else {
            SetUser(null)
            setLoadinguser(false)
        }

    }, [token])

    useEffect(() => {
        if (user) {
            fetchUsersChats()
        } else {
            SetChats([])
            SetSelectedChat(null)
        }
    }, [user])

    useEffect(() => {
        if (theme === 'dark') {
            document.documentElement.classList.add('dark')
        } else {
            document.documentElement.classList.remove('dark')
        }
        localStorage.setItem('theme', theme)
    }, [theme])


    const value = {
        navigate, user, SetUser, fetchUser, chats, SetChats, selectedChat, SetSelectedChat, theme, SetTheme, createNewChat, loadingUser,
        fetchUsersChats, token, setToken, axios
    }

    return (
        <AppContext.Provider value={value}>
            {children}
        </AppContext.Provider>
    )
}
// To access any data from above created context
export const useAppContext = () => useContext(AppContext) 