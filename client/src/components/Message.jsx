import { Clock, Star, User } from 'lucide-react'
import React, { useEffect } from 'react'
import moment from 'moment'
import Markdown from 'react-markdown'
import Prism from 'prismjs'
const Message = ({ message }) => {

  useEffect(()=>{
    Prism.highlightAll()
  },[message.content])
  return (
    <div>
      {
        message.role === "user" ?
          (
            <div className='flex items-start justify-end my-4 gap-2'>
              <div className='flex flex-col gap-2 p-2 px-4 bg-slate-50 dark:bg-neutral-500 border
                border-neutral-400 dark:border-neutral-800 rounded-md max-w-2xl'>
                <p className='text-sm dark:text-white'>{message.content}</p>
                <span className='flex flex-row gap-1 text-xs text-neutral-300'>
                  <Clock className='size-3 mt-0.5' />{moment(message.timestamp).fromNow()}
                </span>
              </div>
              <User className='size-5 bg-neutral-300 dark:bg-neutral-500 rounded-full px-0.5' />
            </div>
          ) :
          (
            <div className='inline-flex flex-col gap-2 p-2 px-4 max-w-2xl bg-slate-50 
            dark:bg-neutral-500 border border-neutral-500 rounded-md my-4'>
              {message.isImage ? (
                <img src={message.content} alt='' className='w-full mt-2 max-w-md rounded-md' />
              ) : (
                <div className='text-sm dark:text-white reset-tw'>
                  <Markdown>
                    {message.content}
                  </Markdown>
                </div>
              )}
              <span className='flex flex-row gap-1 text-xs text-neutral-300'>
                <Clock className='size-3 mt-0.5' />{moment(message.timestamp).fromNow()}
              </span>
            </div>
          )
      }
    </div>

  )
}

export default Message
