import { FILE_TYPE_CONFIG } from '../types'

interface FileTypeIconProps {
  fileName: string
  size?: number
}

function getExtension(fileName: string): string {
  const parts = fileName.split('.')
  return parts.length > 1 ? parts[parts.length - 1].toLowerCase() : ''
}

export function FileTypeIcon({ fileName, size = 32 }: FileTypeIconProps) {
  const extension = getExtension(fileName)
  const config = FILE_TYPE_CONFIG[extension]
  const label = config?.label ?? extension.toUpperCase().slice(0, 3)
  const color = config?.color ?? '#9CA3AF'

  const width = size
  const height = size * 1.25

  return (
    <svg
      width={width}
      height={height}
      viewBox="0 0 32 40"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="shrink-0"
    >
      {/* 파일 형태 (접힌 모서리) */}
      <path
        d="M2 4C2 1.79086 3.79086 0 6 0H20L30 10V36C30 38.2091 28.2091 40 26 40H6C3.79086 40 2 38.2091 2 36V4Z"
        fill="#F3F4F6"
        stroke="#D1D5DB"
        strokeWidth="1"
      />
      <path d="M20 0L30 10H24C21.7909 10 20 8.20914 20 6V0Z" fill="#E5E7EB" />
      {/* 확장자 라벨 배경 */}
      <rect x="1" y="22" width="30" height="14" rx="2" fill={color} />
      {/* 확장자 텍스트 */}
      <text
        x="16"
        y="32.5"
        textAnchor="middle"
        fill="white"
        fontSize="8"
        fontWeight="bold"
        fontFamily="system-ui, sans-serif"
      >
        {label}
      </text>
    </svg>
  )
}
