import type { Metadata } from 'next'
import dynamic from 'next/dynamic'
import { Inter } from 'next/font/google'
import '../styles/globals.css'
import { Providers } from './providers'

const ChatWidget = dynamic(() => import('@/components/ChatWidget'), {
  ssr: false,
})

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: '법률 서비스 플랫폼',
  description: '변호사 추천, 판례 검색, 소액 소송 지원 서비스',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="ko">
      <body className={inter.className}>
        <Providers>
          {children}
          <ChatWidget />
        </Providers>
      </body>
    </html>
  )
}
