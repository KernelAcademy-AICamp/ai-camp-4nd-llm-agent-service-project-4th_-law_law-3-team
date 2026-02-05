---
name: react-nextjs-frontend
description: React/Next.js 프론트엔드 개발 가이드. 컴포넌트 생성, 리팩토링, UI/UX 개선, 접근성 확보, Tailwind CSS 스타일링. FastAPI 백엔드 연동 패턴 포함. 새 컴포넌트 작성, 기존 코드 개선, 접근성 검토 시 사용.
---

# React Next.js Frontend Development

Professional React and Next.js component development with modern best practices, Tailwind CSS styling, and accessibility-first approach.

## Core Principles

1. **Component Architecture**: Build reusable, composable components following React best practices
2. **Accessibility First**: Ensure WCAG 2.1 AA compliance in all UI elements
3. **Type Safety**: Use TypeScript for type safety and better developer experience
4. **Performance**: Optimize for Core Web Vitals and user experience
5. **Consistency**: Maintain consistent code style and patterns across the codebase

## JavaScript Code Style

Follow these conventions for all React/Next.js code:

- Use **single quotes** for strings
- Use **semicolons** at statement ends
- Use **const/let** (never var)
- Use **arrow functions** for components and callbacks
- Use **destructuring** for props and state
- Use **template literals** for string interpolation
- Prefer **named exports** for components (unless default export is specifically needed)
- Use **PascalCase** for component names
- Use **camelCase** for functions, variables, and file names (except components)

## Quick Start Checklist

When starting a new component or feature:

1. Determine the component type (Server/Client component in Next.js)
2. Plan the component hierarchy and data flow
3. Identify accessibility requirements (ARIA labels, keyboard navigation, focus management)
4. Choose appropriate Tailwind utilities for responsive design
5. Consider loading states, error states, and edge cases
6. Plan API integration points if connecting to FastAPI backend

## Component Creation

### File Structure

```
components/
├── ui/              # Reusable UI primitives (Button, Input, Card, etc.)
├── features/        # Feature-specific components
└── layouts/         # Layout components (Header, Footer, Sidebar)

app/                 # Next.js 13+ app directory
├── (routes)/        # Route groups
├── api/             # API routes
└── layout.tsx       # Root layout
```

### Component Template

For detailed component patterns and templates, see `references/component-patterns.md`.

### Naming Conventions

- **Components**: PascalCase (`UserProfile.tsx`, `ProductCard.tsx`)
- **Utilities**: camelCase (`formatDate.ts`, `validateEmail.ts`)
- **Hooks**: camelCase with 'use' prefix (`useAuth.ts`, `useLocalStorage.ts`)
- **Types**: PascalCase with descriptive names (`UserData`, `ApiResponse`)

## Styling with Tailwind CSS

### Core Utilities

Use Tailwind's utility-first approach:

- **Layout**: `flex`, `grid`, `container`, `mx-auto`
- **Spacing**: `p-4`, `mt-6`, `space-y-4`, `gap-2`
- **Typography**: `text-lg`, `font-semibold`, `leading-relaxed`
- **Colors**: `bg-blue-500`, `text-gray-700`, `border-slate-200`
- **Responsive**: `sm:`, `md:`, `lg:`, `xl:`, `2xl:` prefixes
- **Dark mode**: `dark:` prefix for dark mode variants

### Responsive Design

Always design mobile-first, then add responsive breakpoints:

```jsx
<div className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4'>
```

## Accessibility Guidelines

### Essential Practices

1. **Semantic HTML**: Use proper HTML elements (`<button>`, `<nav>`, `<main>`, `<article>`)
2. **ARIA Labels**: Add `aria-label` for icon buttons and non-text elements
3. **Keyboard Navigation**: Ensure all interactive elements are keyboard accessible
4. **Focus Management**: Use visible focus indicators, manage focus on modals/dialogs
5. **Alt Text**: Provide descriptive alt text for images
6. **Color Contrast**: Ensure WCAG AA contrast ratios (4.5:1 for text)

For complete accessibility patterns, see `references/accessibility-guide.md`.

## Backend Integration

### FastAPI Integration

When connecting to FastAPI backends:

```typescript
// api/client.ts
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const apiClient = {
  get: async (endpoint: string) => {
    const res = await fetch(`${API_URL}${endpoint}`);
    if (!res.ok) throw new Error('API Error');
    return res.json();
  },
  // ... other methods
};
```

### Flexible Architecture

Adapt architecture based on backend:

- **FastAPI**: Use RESTful patterns, consider React Query for data fetching
- **GraphQL**: Use Apollo Client or urql
- **tRPC**: Use tRPC client with type safety
- **Firebase**: Use Firebase SDK with custom hooks

See `references/backend-integration.md` for detailed patterns.

## Code Refactoring

When refactoring existing code:

1. **Identify code smells**: Long components, repeated logic, prop drilling
2. **Extract components**: Break large components into smaller, focused ones
3. **Custom hooks**: Extract stateful logic into reusable hooks
4. **Optimize re-renders**: Use `React.memo`, `useMemo`, `useCallback` appropriately
5. **Improve types**: Add or strengthen TypeScript types
6. **Enhance accessibility**: Add missing ARIA attributes and semantic HTML

## UI/UX Improvements

### Common Enhancements

- **Loading states**: Skeleton screens, spinners, progressive loading
- **Error handling**: User-friendly error messages, retry mechanisms
- **Feedback**: Toast notifications, inline validation messages
- **Animations**: Subtle transitions using Tailwind or Framer Motion
- **Empty states**: Helpful messages and CTAs when no data exists
- **Progressive disclosure**: Show advanced options only when needed

### Performance Optimization

- Use Next.js Image component for optimized images
- Implement code splitting and lazy loading
- Optimize bundle size with tree shaking
- Use server components where possible (Next.js 13+)
- Implement proper caching strategies

## Testing Considerations

When creating components, consider:

- Component props and edge cases
- User interactions (clicks, form submissions)
- Accessibility testing (screen readers, keyboard navigation)
- Responsive behavior across breakpoints
- Error states and loading states

## References

- **Component Patterns**: `references/component-patterns.md` - Detailed templates and examples
- **Accessibility Guide**: `references/accessibility-guide.md` - WCAG compliance patterns
- **Backend Integration**: `references/backend-integration.md` - API integration strategies
- **Best Practices**: `references/best-practices.md` - Code quality and optimization tips
