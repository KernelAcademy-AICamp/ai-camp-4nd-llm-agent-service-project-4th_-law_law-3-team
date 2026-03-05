import Link from 'next/link'

export default function NotFound() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 p-4">
      <h2 className="text-xl font-semibold">페이지를 찾을 수 없습니다</h2>
      <p className="text-sm text-gray-500">요청하신 페이지가 존재하지 않습니다.</p>
      <Link
        href="/"
        className="rounded-lg bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700"
      >
        홈으로 돌아가기
      </Link>
    </div>
  )
}
