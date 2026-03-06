import React, { useState } from 'react'
import {useAppContext} from '../context/AppContext'
import {Menu} from 'lucide-react'

const Sidebar = () => {
  // const {chats, setSelectedChat, theme, setTheme, user} = useAppContext()
  const [search, setSearch] = useState('')
  return (
    <div className='flex flex-col h-screen min-w-72 p-5 '>
      <Menu />
      
    </div>
  )
}

export default Sidebar
