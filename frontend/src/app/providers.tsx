'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState } from 'react'
import { UIProvider } from '@/context/UIContext'
import { ChatProvider } from '@/context/ChatContext'

function createQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 1000 * 60 * 5, // 5분
        gcTime: 1000 * 60 * 10, // 10분
        retry: 1,
        refetchOnWindowFocus: false,
      },
    },
  })
}

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(createQueryClient)

  return (
    <QueryClientProvider client={queryClient}>
      <UIProvider>
        <ChatProvider>
          {children}
        </ChatProvider>
      </UIProvider>
    </QueryClientProvider>
  )
}
