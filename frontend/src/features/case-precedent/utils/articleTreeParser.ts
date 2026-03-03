import type { LawArticleItem } from '../types'

export type ArticleNodeType = 'chapter' | 'section' | 'article'

export interface ArticleTreeNode {
  type: ArticleNodeType
  label: string
  article?: LawArticleItem
  children: ArticleTreeNode[]
}

const CHAPTER_RE = /^제\d+장/
const SECTION_RE = /^제\d+절/

/**
 * 조문 배열을 장/절/조 트리 구조로 변환
 *
 * DB에 장/절 컬럼이 없으므로, article_number가 빈 문자열이고
 * article_content에 "제N장"/"제N절" 패턴이 있는 항목을 헤더로 분류
 */
export function buildArticleTree(articles: LawArticleItem[]): ArticleTreeNode[] {
  const roots: ArticleTreeNode[] = []
  let currentChapter: ArticleTreeNode | null = null
  let currentSection: ArticleTreeNode | null = null

  for (const article of articles) {
    const isHeader = !article.article_number || article.article_number.trim() === ''

    if (isHeader) {
      const label = (article.article_title || article.article_content || '').trim()

      if (CHAPTER_RE.test(label)) {
        currentChapter = { type: 'chapter', label, children: [] }
        currentSection = null
        roots.push(currentChapter)
      } else if (SECTION_RE.test(label)) {
        currentSection = { type: 'section', label, children: [] }
        if (currentChapter) {
          currentChapter.children.push(currentSection)
        } else {
          roots.push(currentSection)
        }
      }
      continue
    }

    const node: ArticleTreeNode = {
      type: 'article',
      label: `제${article.article_number}조`,
      article,
      children: [],
    }

    if (currentSection) {
      currentSection.children.push(node)
    } else if (currentChapter) {
      currentChapter.children.push(node)
    } else {
      roots.push(node)
    }
  }

  return roots
}

/** 장/절 구조가 있는 법령인지 판별 */
export function hasTreeStructure(nodes: ArticleTreeNode[]): boolean {
  return nodes.some((n) => n.type === 'chapter' || n.type === 'section')
}

/** 특정 조문번호가 포함된 장/절의 라벨 집합을 반환 (자동 펼침용) */
export function findExpandedLabels(nodes: ArticleTreeNode[], articleNumber: string): Set<string> {
  const labels = new Set<string>()

  function traverse(node: ArticleTreeNode, ancestors: string[]): boolean {
    if (node.type === 'article' && node.article?.article_number === articleNumber) {
      for (const label of ancestors) {
        labels.add(label)
      }
      return true
    }
    for (const child of node.children) {
      if (traverse(child, [...ancestors, node.label])) {
        return true
      }
    }
    return false
  }

  for (const root of nodes) {
    traverse(root, [root.label])
  }
  return labels
}
