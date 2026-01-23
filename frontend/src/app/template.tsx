import '../styles/page-transition.css'

export default function Template({ children }: { children: React.ReactNode }) {
  return (
    <div className="page-transition-wrapper">
      <div className="page-transition-content">
        {children}
      </div>
    </div>
  )
}
