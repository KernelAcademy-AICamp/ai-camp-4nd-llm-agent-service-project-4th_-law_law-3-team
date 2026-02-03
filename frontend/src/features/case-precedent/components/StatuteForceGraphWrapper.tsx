'use client'

import React, { forwardRef } from 'react'
import ForceGraph2D, { type ForceGraphMethods, type ForceGraphProps } from 'react-force-graph-2d'

const ForceGraphWrapper = (props: ForceGraphProps & { graphRef?: React.Ref<ForceGraphMethods> }) => {
  const { graphRef, ...rest } = props
  // 타입 불일치 우회 (LegacyRef vs MutableRefObject)
  return <ForceGraph2D ref={graphRef as any} {...rest} />
}

export default ForceGraphWrapper
