'use client'

import { motion } from 'framer-motion'

export default function Template({ children }: { children: React.ReactNode }) {
  return (
    <div className="w-full h-full" style={{ perspective: '1500px' }}>
      <motion.div
        initial={{ 
          rotateY: -15, 
          opacity: 0, 
          x: 20,
          scale: 0.98,
          transformOrigin: 'left center' 
        }}
        animate={{ 
          rotateY: 0, 
          opacity: 1, 
          x: 0,
          scale: 1,
          transformOrigin: 'left center' 
        }}
        transition={{ 
          duration: 0.8, 
          ease: [0.16, 1, 0.3, 1], // Ease Out Expo for smooth "settling"
          type: "tween" 
        }}
        className="w-full h-full min-h-screen bg-transparent"
        style={{ 
          transformStyle: 'preserve-3d',
          backfaceVisibility: 'hidden'
        }}
      >
        {children}
      </motion.div>
    </div>
  )
}
