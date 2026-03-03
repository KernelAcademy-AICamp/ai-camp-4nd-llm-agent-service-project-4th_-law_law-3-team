import type { Metadata } from 'next'
import dynamic from 'next/dynamic'
import '../styles/globals.css'
import { Providers } from './providers'

const ChatWidget = dynamic(() => import('@/components/ChatWidget'), {
  ssr: false,
})

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
      <head>
        <link
          rel="stylesheet"
          as="style"
          crossOrigin="anonymous"
          href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css"
        />
      </head>
      <body>
        <Providers>
          {children}
          <ChatWidget />
        </Providers>
      </body>
    </html>
  )
}
