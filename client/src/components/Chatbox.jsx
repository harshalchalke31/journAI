import React, { useContext, useEffect, useRef, useState } from 'react'
import { useAppContext } from '../context/AppContext'
import { assets } from '../assets/assets'
import Message from '../components/Message'
import { Dot, Image, Send, SendHorizonal, Text } from 'lucide-react'

const Chatbox = () => {
  // states for chat handling
  const { selectedChat, theme } = useAppContext()
  const [loading, setLoading] = useState(false)
  const [messages, setMessages] = useState([])
  const containerRef = useRef()

  // states for prompt handling
  const [prompt, setPrompt] = useState('')
  const [mode, setMode] = useState('text')
  const [isPublished, setIsPublished] = useState(false)
  const onSubmit = async (e) => {
    e.preventDefault()
  }
  useEffect(() => {
    if (selectedChat) {
      setMessages(selectedChat.messages)
    }
  }, [selectedChat])

  // useeffect for scrolling to latest generated chat
  useEffect(()=>{
    if(containerRef.current){
      containerRef.current.scrollTo({
        top: containerRef.current.scrollHeight,
        behaviour: 'smooth',
      })
    }
  },[messages])
  return (
    <div className='flex-1 flex flex-col justify-between m-5 md:m-10 xl:m-30 max-md:mt-14 2xl:pr-40'>
      {/* Chat messages */}
      <div ref={containerRef} className='flex-1 mb-5 overflow-y-scroll'>
        {
          messages.length === 0 && (
            <div className='h-full flex flex-col items-center justify-center gap-2'>
              <div className='flex items-center gap-3 dark:bg-neutral-800 text-white px-2 py-2 rounded-xl shadow-md w-fit'>
                <img src={assets.logo} className='h-10 w-10 object-contain ' />
                <div className='flex flex-col leading-tight'>
                  <p className='text-lg font-semibold dark:text-white text-black'>RAGBOT</p>
                  <p className='text-xs text-neutral-500'>Enterprise document assistent</p>
                </div>
              </div>
              <p className='mt-5 text-3xl sm:text-4xl text-center text-neutral-400'>Ask me anything ...</p>
            </div>
          )
        }

        {messages.map((message, index) => <Message key={index} message={message} />)}
        {/* Loading animation */}
        {loading &&
          <div className='loader flex items-center gap-1.5'>
            <Dot className='animate-dot-scale size-25 text-neutral-800 dark:text-neutral-100 ' />

          </div>}

      </div>
      {mode==='image'&&(
        <label className='inline-flex items-center gap-2 mb-3 text-sm mx-auto'>
          <p className='text-xs'>Publish Generated images to repository:</p>
          <input type='checkbox' className='cursor-pointer' checked={isPublished}
          onChange={(e)=>setIsPublished(e.target.checked)}/>
        </label>
      )}
      {/* Prompt box */}
      <form onSubmit={onSubmit} className='bg-neutral-300 dark:bg-neutral-800 border border-neutral-500
      dark:border-neutral-600 rounded-xl w-full max-w-2xl p-3 pl-0 mx-auto flex gap-2 items-center'>
        <select onChange={(e) => setMode(e.target.value)} value={mode}
          className='text-sm pl-3 pr-2 outline-none '>
          <option className='dark:bg-neutral-400' value='text'>Text</option>
          <option className='dark:bg-neutral-400' value='image'>Image</option>
        </select>
        <input onChange={(e) => setPrompt(e.target.value)} value={prompt}
          type='text' placeholder='Type your queries here...' required
          className='flex-1 w-full text-sm outline-none' />
        <button disabled={loading}><SendHorizonal className='size-5 cursor-pointer' /></button>
      </form>
    </div>
  )
}

export default Chatbox
