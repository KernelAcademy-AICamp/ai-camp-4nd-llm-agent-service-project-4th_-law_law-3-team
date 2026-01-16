import type { Metadata } from 'next'
import Script from 'next/script'
import { Inter } from 'next/font/google'
import '../styles/globals.css'
import { Providers } from './providers'

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
  const kakaoMapKey = process.env.NEXT_PUBLIC_KAKAO_MAP_API_KEY

  return (
    <html lang="ko">
      <body className={inter.className}>
        {kakaoMapKey && (
          <Script
            src={`https://dapi.kakao.com/v2/maps/sdk.js?appkey=${kakaoMapKey}&libraries=clusterer,services&autoload=false`}
            strategy="beforeInteractive"
          />
        )}
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
