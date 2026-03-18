import React, { useEffect, useState } from 'react'
import { dummyPlans } from '../assets/assets'
import Loading from './Loading'
import { ShoppingBag } from 'lucide-react'

const Credits = () => {
  const [plans, setPlans] = useState([])
  const [loading, setLoading] = useState(true)
  const fetchPlans = async () => {
    setPlans(dummyPlans)
    setLoading(false)
  }

  useEffect(() => {
    fetchPlans()
  }, [])

  if (loading) return <Loading />
  return (
    <div className='max-w-7xl h-screen overflow-y-scroll mx-auto px-4 sm:px-6 lg:px-8 py-12'>
      <h2 className='text-3xl font-semibold text-center mb-10 xl:mt-30 text-gray-800
      dark:text-white'>Credit Plans</h2>
      <div className='flex flex-wrap justify-center gap-8'>
        {plans.map((plan) => (
          <div key={plan._id} className={`border border-gray-600 dark:border-gray-200 rounded-xl
          shadow hover:shadow-lg transition-shadow p-6 min-w-[300px] flex flex-col
          ${plan._id === 'pro' ? 'bg-white dark:bg-gray-200' : 'bg-gray-200 dark:bg-gray-400'}`}>

            <div className='flex-1'>
              <h3 className='text-xl font-semibold text-gray-600  mb-2'>{plan.name}</h3>
              <p className='text-2xl font-bold text-gray-900 mb-4'>${plan.price}
                <span className='text-base font-normal text-gray-600
                '>{' '}/ {plan.credits} credits</span>
              </p>
              <ul className='list-disc list-inside text-sm text-gray-700 space-y-1'>
                {plan.features.map((feature, index)=>(
                  <li key={index}>{feature}</li>
                ))}
              </ul>
            </div>
            <button className='mt-6 bg-gray-400 dark:bg-white dark:text-gray-600 active:bg-black
            text-white font-medium py-2 rounded transition-colors cursor-pointer flex flex-row gap-2
            justify-center'>
              <ShoppingBag />Buy Now
              </button>
          </div>
        ))}

      </div>
    </div>
  )
}

export default Credits
