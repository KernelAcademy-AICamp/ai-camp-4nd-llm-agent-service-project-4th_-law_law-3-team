// doc_type을 한글로 변환
export const docTypeLabels: Record<string, string> = {
  precedent: '판례',
  constitutional: '헌재결정',
  administration: '행정심판',
  legislation: '입법예고',
  committee: '위원회결정',
  law: '법령',
}

export function getDocTypeLabel(docType: string): string {
  return docTypeLabels[docType] || docType
}

// doc_type별 배지 색상
export function getDocTypeBadgeColor(docType: string): string {
  switch (docType) {
    case 'law':
      return 'bg-green-50 text-green-600'
    case 'precedent':
      return 'bg-blue-50 text-blue-600'
    case 'constitutional':
      return 'bg-purple-50 text-purple-600'
    case 'committee':
      return 'bg-orange-50 text-orange-600'
    case 'administration':
      return 'bg-yellow-50 text-yellow-700'
    case 'legislation':
      return 'bg-teal-50 text-teal-600'
    default:
      return 'bg-gray-50 text-gray-600'
  }
}
