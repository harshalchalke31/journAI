import React, { useState } from 'react'
import Sidebar from './components/Sidebar'
import { Routes, Route, useLocation } from 'react-router-dom'
import Chatbox from './components/Chatbox'
import Credits from './pages/Credits'
import Community from './pages/Community'
import { Menu } from 'lucide-react'
import './assets/prism.css'
import Loading from './pages/Loading'
import { useAppContext } from './context/AppContext'
import Login from './pages/Login'
import {Toaster} from 'react-hot-toast'


const App = () => {
  const { user, loadingUser } = useAppContext()
  const [isMenuOpen, SetIsMenuOpen] = useState(false)
  const { pathname } = useLocation()
  if (pathname === '/loading' || loadingUser) return <Loading />
  return (
    <>
    <Toaster />
      {!isMenuOpen && <Menu className='absolute top-3 left-3 size-6 
    dark:hover:bg-neutral-500 hover:bg-neutral-300 rounded-md 
    cursor-pointer md:hidden dark:text-white'
        onClick={() => SetIsMenuOpen(true)} />}

      {user ? (
        <div className='dark:bg-linear-to-b from-[#242124] to-[#000000] dark:text-white'>
          <div className='flex h-screen w-screen'>
            <Sidebar isMenuOpen={isMenuOpen} SetIsMenuOpen={SetIsMenuOpen} />
            <Routes>
              <Route path='/' element={<Chatbox />} />
              <Route path='/credits' element={<Credits />} />
              <Route path='/community' element={<Community />} />
            </Routes>
          </div>
        </div>
      ) : (
        <div className='bg-neutral-800 flex items-center justify-center h-screen w-screen'>
          <Login />
        </div>

      )}


    </>
  )
}

export default App
