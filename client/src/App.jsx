import React from 'react'
import Sidebar from './components/Sidebar'
import { Routes, Route } from 'react-router-dom'
import Chatbox from './components/Chatbox'
import Credits from './pages/Credits'
import Community from './pages/Community'


const App = () => {
  return (
    <div className='dark:bg-linear-to-b from-[#242124] to-[#000000] dark:text-white'>
      <div className='flex h-screen w-screen'>
        <Sidebar />
        <Routes>
          <Route path='/' element={<Chatbox />} />
          <Route path='/credits' element={<Credits />} />
          <Route path='/community' element={<Community />} />
        </Routes>
      </div>
    </div>
  )
}

export default App
