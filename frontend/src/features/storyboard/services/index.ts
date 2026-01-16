import { api, endpoints } from '@/lib/api'

interface StoryboardEvent {
  date: string
  description: string
}

export const storyboardService = {
  generateStoryboard: async (title: string, events: StoryboardEvent[]) => {
    const response = await api.post(`${endpoints.storyboard}/generate`, {
      title,
      events,
    })
    return response.data
  },

  getStoryboard: async (storyboardId: string) => {
    const response = await api.get(`${endpoints.storyboard}/${storyboardId}`)
    return response.data
  },

  regeneratePanel: async (
    storyboardId: string,
    panelIndex: number,
    newPrompt: string
  ) => {
    const response = await api.post(
      `${endpoints.storyboard}/${storyboardId}/regenerate-panel`,
      { panel_index: panelIndex, new_prompt: newPrompt }
    )
    return response.data
  },
}
