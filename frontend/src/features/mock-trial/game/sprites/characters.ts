/** 5종 캐릭터 설정 */

import { CHARACTER_COLORS, CHARACTER_NAMES, CHARACTER_POSITIONS } from '../config'

export interface CharacterConfig {
  role: string
  name: string
  color: number
  position: { x: number; y: number }
  palette?: Record<string, number>
}

export const CHARACTERS: CharacterConfig[] = [
  {
    role: 'judge',
    name: CHARACTER_NAMES.judge,
    color: CHARACTER_COLORS.judge,
    position: CHARACTER_POSITIONS.judge,
  },
  {
    role: 'prosecutor',
    name: CHARACTER_NAMES.prosecutor,
    color: CHARACTER_COLORS.prosecutor,
    position: CHARACTER_POSITIONS.prosecutor,
  },
  {
    role: 'attorney',
    name: CHARACTER_NAMES.attorney,
    color: CHARACTER_COLORS.attorney,
    position: CHARACTER_POSITIONS.attorney,
  },
  {
    role: 'defendant',
    name: CHARACTER_NAMES.defendant,
    color: CHARACTER_COLORS.defendant,
    position: CHARACTER_POSITIONS.defendant,
  },
  {
    role: 'clerk',
    name: CHARACTER_NAMES.clerk,
    color: CHARACTER_COLORS.clerk,
    position: CHARACTER_POSITIONS.clerk,
  },
]
