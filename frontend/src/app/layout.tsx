import type { Metadata } from 'next'
import dynamic from 'next/dynamic'
import { Suspense } from 'react'
import '../styles/globals.css'
import { Providers } from './providers'
import Sidebar from '@/components/Sidebar'
import LayoutWrapper from '@/components/LayoutWrapper'

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
          rel="preload"
          as="style"
          crossOrigin="anonymous"
          href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css"
        />
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){var l=document.createElement("link");l.rel="stylesheet";l.crossOrigin="anonymous";l.href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css";document.head.appendChild(l)})()`,
          }}
        />
        <noscript>
          <link
            rel="stylesheet"
            crossOrigin="anonymous"
            href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css"
          />
        </noscript>
      </head>
      <body>
        <Providers>
          <div className="flex min-h-screen relative">
            <Suspense fallback={null}>
              <Sidebar />
            </Suspense>
            <LayoutWrapper>
              {children}
            </LayoutWrapper>
          </div>
          <ChatWidget />
        </Providers>
      </body>
    </html>
  )
}
