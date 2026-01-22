/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * 카카오맵 SDK 타입 정의
 */

declare namespace kakao.maps {
  // 로드 함수
  function load(callback: () => void): void

  // 지도 클래스
  class Map {
    constructor(container: HTMLElement, options: MapOptions)
    setCenter(latlng: LatLng): void
    getCenter(): LatLng
    setLevel(level: number): void
    getLevel(): number
    getBounds(): LatLngBounds
    panTo(latlng: LatLng): void
    setMapTypeId(mapTypeId: MapTypeId): void
    relayout(): void
  }

  interface MapOptions {
    center: LatLng
    level?: number
    mapTypeId?: MapTypeId
    draggable?: boolean
    scrollwheel?: boolean
    disableDoubleClick?: boolean
    disableDoubleClickZoom?: boolean
  }

  // 좌표 클래스
  class LatLng {
    constructor(lat: number, lng: number)
    getLat(): number
    getLng(): number
    equals(latlng: LatLng): boolean
    toString(): string
  }

  // 좌표 경계
  class LatLngBounds {
    constructor(sw?: LatLng, ne?: LatLng)
    extend(latlng: LatLng): void
    getSouthWest(): LatLng
    getNorthEast(): LatLng
    isEmpty(): boolean
    toString(): string
  }

  // 마커
  class Marker {
    constructor(options: MarkerOptions)
    setMap(map: Map | null): void
    getMap(): Map | null
    setPosition(latlng: LatLng): void
    getPosition(): LatLng
    setImage(image: MarkerImage): void
    setTitle(title: string): void
    setDraggable(draggable: boolean): void
    setClickable(clickable: boolean): void
    setZIndex(zIndex: number): void
  }

  interface MarkerOptions {
    position: LatLng
    map?: Map
    image?: MarkerImage
    title?: string
    draggable?: boolean
    clickable?: boolean
    zIndex?: number
  }

  // 마커 이미지
  class MarkerImage {
    constructor(src: string, size: Size, options?: MarkerImageOptions)
  }

  interface MarkerImageOptions {
    offset?: Point
    alt?: string
    coords?: string
    shape?: string
    spriteOrigin?: Point
    spriteSize?: Size
  }

  // 크기
  class Size {
    constructor(width: number, height: number)
  }

  // 포인트
  class Point {
    constructor(x: number, y: number)
  }

  // 원
  class Circle {
    constructor(options: CircleOptions)
    setMap(map: Map | null): void
    setPosition(latlng: LatLng): void
    setRadius(radius: number): void
    setOptions(options: Partial<CircleOptions>): void
  }

  interface CircleOptions {
    center: LatLng
    radius: number
    strokeWeight?: number
    strokeColor?: string
    strokeOpacity?: number
    strokeStyle?: string
    fillColor?: string
    fillOpacity?: number
    map?: Map
  }

  // 커스텀 오버레이
  class CustomOverlay {
    constructor(options: CustomOverlayOptions)
    setMap(map: Map | null): void
    getMap(): Map | null
    setPosition(latlng: LatLng): void
    getPosition(): LatLng
    setContent(content: string | HTMLElement): void
    setZIndex(zIndex: number): void
  }

  interface CustomOverlayOptions {
    map?: Map
    position: LatLng
    content: string | HTMLElement
    xAnchor?: number
    yAnchor?: number
    zIndex?: number
    clickable?: boolean
  }

  // 인포윈도우
  class InfoWindow {
    constructor(options: InfoWindowOptions)
    open(map: Map, marker?: Marker): void
    close(): void
    setContent(content: string | HTMLElement): void
    setPosition(latlng: LatLng): void
  }

  interface InfoWindowOptions {
    content?: string | HTMLElement
    position?: LatLng
    removable?: boolean
    zIndex?: number
    disableAutoPan?: boolean
  }

  // 이벤트
  namespace event {
    function addListener(target: any, type: string, callback: (e?: any) => void): void
    function removeListener(target: any, type: string, callback: (e?: any) => void): void
    function trigger(target: any, type: string, data?: any): void
  }

  // 지도 타입
  const enum MapTypeId {
    ROADMAP = 1,
    SKYVIEW = 2,
    HYBRID = 3,
  }

  // 마커 클러스터러
  class MarkerClusterer {
    constructor(options: MarkerClustererOptions)
    addMarker(marker: Marker, nodraw?: boolean): void
    addMarkers(markers: Marker[], nodraw?: boolean): void
    removeMarker(marker: Marker, nodraw?: boolean): void
    removeMarkers(markers: Marker[], nodraw?: boolean): void
    clear(): void
    redraw(): void
    getMap(): Map
    setMap(map: Map | null): void
    setGridSize(size: number): void
    getGridSize(): number
    setMinClusterSize(size: number): void
    getMinClusterSize(): number
    setAverageCenter(bool: boolean): void
    getAverageCenter(): boolean
    setMinLevel(level: number): void
    getMinLevel(): number
    setTexts(texts: string[] | ((size: number) => string)): void
    getTexts(): string[]
    setCalculator(calculator: (size: number) => number): void
    getCalculator(): (size: number) => number
    setStyles(styles: object[]): void
    getStyles(): object[]
  }

  interface MarkerClustererOptions {
    map: Map
    markers?: Marker[]
    gridSize?: number
    averageCenter?: boolean
    minLevel?: number
    minClusterSize?: number
    disableClickZoom?: boolean
    styles?: object[]
    texts?: string[] | ((size: number) => string)
    calculator?: (size: number) => number
  }

  // 지오코더 서비스
  namespace services {
    class Geocoder {
      addressSearch(address: string, callback: (result: any[], status: Status) => void): void
      coord2Address(lng: number, lat: number, callback: (result: any[], status: Status) => void): void
      coord2RegionCode(lng: number, lat: number, callback: (result: any[], status: Status) => void): void
    }

    class Places {
      keywordSearch(keyword: string, callback: (result: any[], status: Status, pagination: any) => void, options?: object): void
      categorySearch(code: string, callback: (result: any[], status: Status, pagination: any) => void, options?: object): void
      setMap(map: Map): void
    }

    const enum Status {
      OK = 'OK',
      ZERO_RESULT = 'ZERO_RESULT',
      ERROR = 'ERROR',
    }
  }
}

interface Window {
  kakao: typeof kakao
}
