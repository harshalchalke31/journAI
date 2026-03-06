import { createContext, useContext, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {dummyUserData,  dummyChats } from '../assets/assets'

const AppContext = createContext()
export const AppContextProvider = ({children})=>{

    const navigate = useNavigate()
    const [user, SetUser] = useState(null)
    const [chats, SetChats] = useState([])
    const [selectedChat, SetSelectedChat] = useState(null)
    const [theme, SetTheme] = useState(localStorage.getItem('theme') || 'light')
    const fetchUser = async()=>{
        SetUser(dummyUserData)
    }

    const fetchUsersChats = async() =>{
        SetChats(dummyChats)
        SetSelectedChat(dummyChats[0])
    }


    useEffect(()=>{
        fetchUser()
    },[])
    useEffect(()=>{
        if (user){
            fetchUsersChats()
        } else{
            SetChats([])
            SetSelectedChat(null)
        }
    },[user])

    useEffect(()=>{
        if(theme==='dark'){
            document.documentElement.classList.add('dark')
        }else{
            document.documentElement.classList.remove('dark')
        }
        localStorage.setItem('theme', theme)
    },[theme])


    const value={
        navigate, user, SetUser, fetchUser, chats, SetChats, selectedChat, SetSelectedChat, theme, SetTheme
    }

    return (
        <AppContext.Provider value={value}>
            {children}
        </AppContext.Provider>
    )
}
// To access any data from above created context
export const useAppContext = ()=> useContext(AppContext) 