/** Phaser <-> React 이벤트 통신 버스 */

import type { EvidenceItem, EmotionType } from '../types'

export interface EventMap {
  // Phaser -> React
  'agent:speak': { agent: string; text: string; streaming: boolean; emotion?: EmotionType }
  'stage:change': { from: string; to: string; stageNumber: number; totalStages: number }
  'evidence:presented': { cases: EvidenceItem[]; articles: EvidenceItem[] }
  'trial:complete': { judgment: string; feedback: string }
  'game:ready': Record<string, never>

  // React -> Phaser
  'user:input': { text: string }
  'user:select_evidence': { evidenceIds: string[] }
  'game:advance_stage': Record<string, never>
  'agent:animate': { agent: string; animation: 'idle' | 'speak' | 'react' }
  'setup:complete': { caseType: string; userRole: string; caseSummary: string }
}

class CourtEventBus {
  private target = new EventTarget()

  emit<K extends keyof EventMap>(event: K, data: EventMap[K]): void {
    this.target.dispatchEvent(new CustomEvent(event, { detail: data }))
  }

  on<K extends keyof EventMap>(event: K, handler: (data: EventMap[K]) => void): () => void {
    const listener = (e: Event): void => {
      handler((e as CustomEvent).detail as EventMap[K])
    }
    this.target.addEventListener(event, listener)
    return () => this.target.removeEventListener(event, listener)
  }
}

export const eventBus = new CourtEventBus()
