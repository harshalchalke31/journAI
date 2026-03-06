import React, { useState } from 'react'
import { useAppContext } from '../context/AppContext'
import { Coins, Cross, CrossIcon, Diamond, DiamondIcon, DiamondPlus, Edit, File, GalleryHorizontal, GalleryThumbnails, Image, Images, LogOut, PictureInPicture, Search, Trash2, Upload, User, X } from 'lucide-react'
import { assets } from '../assets/assets'
import moment from 'moment'

const Sidebar = ({isMenuOpen, SetIsMenuOpen}) => {
  const { chats, setSelectedChat, theme, SetTheme, user, navigate } = useAppContext()
  const [search, setSearch] = useState('')
  return (
    // bg-gradient-to-b from-[#242124]/30 to-[#000000]/30
    <div className={`flex flex-col h-screen min-w-72 p-5 dar:bg-neutral-700
    border-r border-[#80609F]/30 backdrop-blur-3xl transition-all duration-500 max-md:absolute left-0 z-1
    ${!isMenuOpen && 'max-md:-translate-x-full'}`}>
      {/* Logo */}
      <div className='flex items-center gap-3 dark:bg-neutral-800 text-white px-2 py-2 rounded-xl
      shadow-md w-fit'>
        <img src={assets.logo} className='h-10 w-10 object-contain ' />
        <div className='flex flex-col leading-tight'>
          <p className='text-lg font-semibold dark:text-white text-black'>RAGBOT</p>
          <p className='text-xs text-neutral-500'>Enterprise document assistent</p>
        </div>
      </div>
      {/* Neww chat button */}
      <button className='flex gap-2 items-center w-full py-2 px-2 mt-2 dark:text-white text-black dark:bg-neutral-700
       bg-neutral-200 text-sm rounded-xl cursor-pointer hover:bg-neutral-300 dark:hover:bg-neutral-500 transition-colors duration-300'>
        <Edit className='size-5' />New Chat
      </button>
      <button className='flex gap-2 items-center w-full py-2 px-2 mt-2 dark:text-white text-black dark:bg-neutral-700
       bg-neutral-200 text-sm rounded-xl cursor-pointer hover:bg-neutral-300 dark:hover:bg-neutral-500 transition-colors duration-300'>
        <Upload className='size-5' />Upload
      </button>
      <button className='flex gap-2 items-center w-full py-2 px-2 mt-2 dark:text-white text-black dark:bg-neutral-700
       bg-neutral-200 text-sm rounded-xl cursor-pointer hover:bg-neutral-300 dark:hover:bg-neutral-500 transition-colors duration-300'>
        <File className='size-5' />Enterprise Documents
      </button>
      {/* Search bar */}
      <div className='flex items-center gap-2 p-3 mt-2 border border-gray-400 dark:border-white/20 rounded-xl'>
        <Search className='size-5 dark:text-white' />
        <input placeholder='Search conversations...' onChange={(e) => setSearch(e.target.value)} value={search}
          className='text-xs placeholder:text-neutral-400 outline-none' />
      </div>
      {/* Recent Chats */}
      {chats.length > 0 && <p className='mt-3 text-sm'>Recent Chats</p>}
      <div className='flex-1 overflow-y-scroll mt-2 text-sm space-y-3'>
        {chats.filter((chat) => chat.messages[0] ? chat.messages[0]?.content.toLowerCase()
          .includes(search.toLowerCase()) : chat.name.toLowerCase().includes(search.toLowerCase()))
          .map((chat) => (
            <div onClick={()=>{navigate('/'); SetIsMenuOpen(false);setSelectedChat(chat); }}
            key={chat._id} className='p-2 px-4 dark:bg-[#57317C]/10 border border-neutral-500
          dark:border-[#80609F]/15 rounded-xl cursor-pointer flex justify-between group'>
              <div>
                <p className='truncate-w-full'>{chat.messages.length > 0 ? chat.messages[0].content.slice(0, 32) :
                  chat.name}</p>
                <p className='text-xs text-neutral-500 dark:text-[#B1A6C0]'>{moment(chat.updatedAt).fromNow()}</p>
              </div>
              <Trash2 className='hidden group-hover:block w-4 cursor-pointer dark:invert' />

            </div>
          ))}
      </div>
      {/* Community Images */}
      <div onClick={() => { navigate('/community');SetIsMenuOpen(false) }} className='flex items-center gap-2 p-3 mt-2 border border-neutral-500 rounded-xl
      dark:border-neutral-300 cursor-pointer hover:scale-105 transition-all duration-300'>
        <Images className='size-5 dark:text-white' />
        <p className='text-sm'>Generated Images</p>
      </div>
      {/* Credit page */}
      <div onClick={() => { navigate('/credits');SetIsMenuOpen(false) }} className='flex items-center gap-2 p-3 mt-2 border border-neutral-500 rounded-xl
      dark:border-neutral-300 cursor-pointer hover:scale-105 transition-all duration-300'>
        <DiamondPlus className='size-5 dark:text-white' />
        <div className='flex flex-col text-sm'>
          <p className='text-sm'>Credits: {user?.credits}</p>
          <p className='text-xs text-gray-400'>Purchase Credits</p>
        </div>
      </div>

      {/* Dark mode toggle */}
      <div className='flex items-center justify-between gap-2 p-3 mt-2 border border-neutral-500 rounded-xl
      dark:border-neutral-300'>
        <div className='flex items-center gap-2 text-sm'>
          <Image className='size-5 dark:text-white' />
          <p className=''>Dark Mode</p>
        </div>
        {/* Toggle Button */}
        <label className='relative inline-flex cursor-pointer'>
          <input onChange={() => SetTheme(theme === 'dark' ? 'light' : 'dark')}
            type='checkbox' className='sr-only peer' checked={theme === 'dark'} />
          <div className='w-9 h-5 bg-gray-400 rounded-full peer-checked:bg-gray-400 transition-all' />
          <span className='absolute left-1 top-1 w-3 h-3 bg-white rounded-full transition-transform
          peer-checked:translate-x-4'/>
        </label>
      </div>
      {/* User Account */}
      <div className='flex items-center gap-2 p-3 mt-2 border border-neutral-500 rounded-xl
      dark:border-neutral-300 cursor-pointer group'>
        <User className='size-5 dark:text-white rounded-full' />
        <p className='flex-1 text-sm dark:text-white truncate'>{user ? user.name : 'Login'}</p>
        {user && <LogOut className='size-5 cursor-pointer hidden dark:text-white group-hover:block' />}
      </div>
          <X onClick={()=>SetIsMenuOpen(false)}
          className='absolute top-3 right-3 size-5 cursor-pointer md:hidden dark:text-white 
          dark:hover:bg-neutral-500 hover:bg-neutral-300 rounded-md'/>
    </div>
  )
}

export default Sidebar
