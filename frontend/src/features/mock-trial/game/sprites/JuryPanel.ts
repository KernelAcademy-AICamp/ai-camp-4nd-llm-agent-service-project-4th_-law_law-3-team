/** 배심원단 관리 + 발언 반응 로직 */

import Phaser from 'phaser'
import { JURY_POSITIONS, JUROR_FACING } from '../config'
import { JurorSprite } from './JurorSprite'
import type { JurorReaction } from './JurorSprite'

/** 역할별 반응 확률과 반응 타입 가중치 */
interface ReactionRule {
  probability: number
  reactions: JurorReaction[]
}

const REACTION_RULES: Record<string, ReactionRule> = {
  judge: { probability: 0.5, reactions: ['neutral', 'nod'] },
  prosecutor: { probability: 0.75, reactions: ['nod', 'think', 'surprise'] },
  attorney: { probability: 0.75, reactions: ['think', 'nod', 'surprise'] },
  defendant: { probability: 0.75, reactions: ['think', 'nod'] },
  clerk: { probability: 0.25, reactions: ['neutral'] },
}

/** 키워드 기반 강한 반응 트리거 */
const STRONG_KEYWORDS: Record<string, string[]> = {
  prosecutor: ['유죄', '징역', '구형', '범행', '증거'],
  attorney: ['무죄', '반박', '증거 불충분', '석방', '변론'],
  defendant: ['반성', '후회', '용서', '죄송', '잘못'],
}

const REACTION_DELAY_MIN = 300
const REACTION_DELAY_MAX = 800
const REACTION_DURATION = 2500

export class JuryPanel {
  private scene: Phaser.Scene
  private jurors: JurorSprite[] = []
  private resetTimers: Phaser.Time.TimerEvent[] = []

  constructor(scene: Phaser.Scene) {
    this.scene = scene

    JURY_POSITIONS.forEach((pos, index) => {
      const juror = new JurorSprite(scene, pos.x, pos.y, index, JUROR_FACING)
      this.jurors.push(juror)
    })
  }

  reactToSpeech(agent: string, text: string): void {
    this.clearResetTimers()

    const rule = REACTION_RULES[agent] ?? REACTION_RULES.clerk
    const hasStrongKeyword = this.containsStrongKeyword(agent, text)
    const effectiveProbability = hasStrongKeyword
      ? Math.min(rule.probability + 0.2, 1.0)
      : rule.probability

    this.jurors.forEach((juror) => {
      const shouldReact = Math.random() < effectiveProbability
      if (!shouldReact) return

      const delay = REACTION_DELAY_MIN +
        Math.random() * (REACTION_DELAY_MAX - REACTION_DELAY_MIN)

      const reaction = this.pickReaction(rule.reactions, hasStrongKeyword)

      this.scene.time.delayedCall(delay, () => {
        juror.react(reaction)
      })

      const resetTimer = this.scene.time.delayedCall(
        delay + REACTION_DURATION,
        () => {
          juror.stopReaction()
        }
      )
      this.resetTimers.push(resetTimer)
    })
  }

  private containsStrongKeyword(agent: string, text: string): boolean {
    const keywords = STRONG_KEYWORDS[agent]
    if (!keywords) return false
    return keywords.some((keyword) => text.includes(keyword))
  }

  private pickReaction(
    reactions: JurorReaction[],
    isStrong: boolean
  ): JurorReaction {
    if (isStrong && reactions.length > 1) {
      const nonNeutral = reactions.filter((r) => r !== 'neutral')
      if (nonNeutral.length > 0) {
        return nonNeutral[Math.floor(Math.random() * nonNeutral.length)]
      }
    }
    return reactions[Math.floor(Math.random() * reactions.length)]
  }

  private clearResetTimers(): void {
    this.resetTimers.forEach((timer) => timer.destroy())
    this.resetTimers = []
    this.jurors.forEach((juror) => juror.stopReaction())
  }

  destroy(): void {
    this.clearResetTimers()
    this.jurors.forEach((juror) => juror.destroy())
    this.jurors = []
  }
}
