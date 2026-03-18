import { Loader, Loader2 } from 'lucide-react'
import React, { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

const Loading = () => {
  const navigate = useNavigate()
  useEffect(()=>{
    const timeout = setTimeout(()=>{
      navigate('/')
    },8000)
    return ()=>clearTimeout(timeout)
  },[])
  return (
    <div className='dark:bg-neutral-700 bg-neutral-400 backdrop-opacity-60 flex
    items-center justify-center h-screen w-screen text-black dark:text-white text-2xl'>
      <Loader2 className='size-10 animate-spin'  />
    
    </div>
  )
}

export default Loading
