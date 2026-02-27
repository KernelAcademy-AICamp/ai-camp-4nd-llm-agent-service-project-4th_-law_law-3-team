/* eslint-disable */
/**
 * react-plotly.js 타입 선언 (공식 @types 패키지 미제공으로 직접 선언)
 */
declare module 'react-plotly.js' {
  import { Component } from 'react'

  type PlotType = 'scatter' | 'bar' | 'heatmap' | 'pie' | 'surface' | 'mesh3d' | string
  type PlotMode = 'lines' | 'markers' | 'text' | 'lines+markers' | 'markers+text' | 'lines+text' | 'lines+markers+text' | 'none'
  type HoverInfo = 'x' | 'y' | 'z' | 'text' | 'name' | 'all' | 'none' | 'skip' | string

  interface PlotMarker {
    size?: number | number[]
    color?: string | string[]
    opacity?: number | number[]
    symbol?: string | string[]
    line?: { color?: string | string[]; width?: number | number[] }
    colorscale?: Array<[number, string]>
    showscale?: boolean
    sizemode?: 'diameter' | 'area'
    sizeref?: number
    sizemin?: number
  }

  interface PlotData {
    type?: PlotType
    mode?: PlotMode
    x?: Array<string | number>
    y?: Array<string | number>
    z?: Array<Array<number>>
    text?: string | string[]
    hovertext?: string | string[]
    hoverinfo?: HoverInfo
    hovertemplate?: string | string[]
    name?: string
    marker?: PlotMarker
    textposition?: string
    textfont?: { size?: number; color?: string }
    opacity?: number
    colorscale?: Array<[number, string]>
    showscale?: boolean
    customdata?: Array<number | string | object>
    showlegend?: boolean
  }

  interface LayoutAxis {
    title?: { text?: string; font?: { size?: number }; standoff?: number }
    categoryorder?: 'trace' | 'category ascending' | 'category descending' | 'array' | 'total ascending' | 'total descending' | 'min ascending' | 'min descending' | 'max ascending' | 'max descending' | 'sum ascending' | 'sum descending' | 'mean ascending' | 'mean descending' | 'median ascending' | 'median descending'
    categoryarray?: string[]
    tickfont?: { size?: number }
    tickangle?: number
    gridcolor?: string
  }

  interface PlotShape {
    type?: 'rect' | 'circle' | 'line' | 'path'
    xref?: string
    yref?: string
    x0?: string | number
    x1?: string | number
    y0?: number
    y1?: number
    fillcolor?: string
    line?: { width?: number; color?: string }
    layer?: 'below' | 'above'
  }

  interface PlotLegend {
    orientation?: 'v' | 'h'
    x?: number
    y?: number
    xanchor?: 'left' | 'center' | 'right' | 'auto'
    yanchor?: 'top' | 'middle' | 'bottom' | 'auto'
    font?: { size?: number }
  }

  interface PlotLayout {
    autosize?: boolean
    width?: number
    height?: number
    margin?: { l?: number; r?: number; t?: number; b?: number }
    xaxis?: LayoutAxis
    yaxis?: LayoutAxis
    plot_bgcolor?: string
    paper_bgcolor?: string
    showlegend?: boolean
    legend?: PlotLegend
    hovermode?: 'x' | 'y' | 'closest' | false | 'x unified' | 'y unified'
    annotations?: object[]
    shapes?: PlotShape[]
    transition?: { duration?: number; easing?: string }
    title?: string | { text?: string }
  }

  interface PlotConfig {
    responsive?: boolean
    displayModeBar?: boolean | 'hover'
    modeBarButtonsToRemove?: string[]
    displaylogo?: boolean
    scrollZoom?: boolean
  }

  interface PlotParams {
    data: PlotData[]
    layout?: PlotLayout
    config?: PlotConfig
    style?: React.CSSProperties
    className?: string
    onHover?: (event: object) => void
    onClick?: (event: object) => void
    onRelayout?: (event: object) => void
    useResizeHandler?: boolean
    revision?: number
  }

  export type { PlotData, PlotLayout, PlotConfig, PlotMarker, PlotParams }
  export default class Plot extends Component<PlotParams> {}
}
