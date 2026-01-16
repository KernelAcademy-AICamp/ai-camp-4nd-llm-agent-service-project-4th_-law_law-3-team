# Best Practices

Code quality guidelines and optimization techniques for React/Next.js applications.

## Component Design Principles

### Single Responsibility Principle

Each component should have one clear purpose:

```typescript
// Bad: Component doing too many things
const UserDashboard = () => {
  // Fetching user data
  // Rendering profile
  // Handling settings
  // Managing notifications
  // ...
};

// Good: Separated concerns
const UserDashboard = () => (
  <div>
    <UserProfile />
    <UserSettings />
    <NotificationCenter />
  </div>
);
```

### Composition Over Inheritance

Favor composition patterns:

```typescript
// Good: Composition with children
export const Card = ({ children }: { children: ReactNode }) => (
  <div className='border rounded-lg p-4'>{children}</div>
);

export const CardHeader = ({ children }: { children: ReactNode }) => (
  <div className='border-b pb-2 mb-2'>{children}</div>
);

// Usage
<Card>
  <CardHeader>
    <h2>Title</h2>
  </CardHeader>
  <p>Content</p>
</Card>
```

### Props Destructuring

Always destructure props for clarity:

```typescript
// Good
export const Button = ({ label, onClick, disabled }: ButtonProps) => {
  return <button onClick={onClick} disabled={disabled}>{label}</button>;
};

// Avoid
export const Button = (props: ButtonProps) => {
  return <button onClick={props.onClick} disabled={props.disabled}>{props.label}</button>;
};
```

## State Management Best Practices

### Use the Right State Solution

- **Local state** (`useState`): Component-specific data
- **Lifted state**: Shared between few components
- **Context**: Theme, auth, rarely changing data
- **External libraries** (Zustand, Redux): Complex global state

```typescript
// Local state for component-specific data
const [count, setCount] = useState(0);

// Context for app-wide settings
const ThemeContext = createContext<'light' | 'dark'>('light');

// External state for complex app state
import { create } from 'zustand';

const useStore = create((set) => ({
  users: [],
  addUser: (user) => set((state) => ({ users: [...state.users, user] })),
}));
```

### Avoid Prop Drilling

Use composition or context instead of passing props through many levels:

```typescript
// Bad: Prop drilling
const App = () => {
  const user = useUser();
  return <Layout user={user} />;
};

const Layout = ({ user }) => <Content user={user} />;
const Content = ({ user }) => <Profile user={user} />;

// Good: Context
const UserContext = createContext(null);

const App = () => {
  const user = useUser();
  return (
    <UserContext.Provider value={user}>
      <Layout />
    </UserContext.Provider>
  );
};

const Profile = () => {
  const user = useContext(UserContext);
  return <div>{user.name}</div>;
};
```

## Performance Optimization

### Memoization

Use memoization strategically, not everywhere:

```typescript
// Good: Memoize expensive calculations
const expensiveValue = useMemo(() => {
  return heavyComputation(data);
}, [data]);

// Good: Memoize callback to prevent re-renders
const handleClick = useCallback(() => {
  doSomething(id);
}, [id]);

// Bad: Over-memoization
const trivialValue = useMemo(() => x + y, [x, y]); // Just compute it directly
```

### React.memo for Pure Components

```typescript
// Memoize components that render the same output for same props
export const ExpensiveComponent = memo(({ data }: { data: string }) => {
  console.log('Rendering ExpensiveComponent');
  return <div>{data}</div>;
});

// Don't memoize simple components
export const SimpleText = ({ text }: { text: string }) => <p>{text}</p>;
```

### Code Splitting

```typescript
// Dynamic imports for large components
import dynamic from 'next/dynamic';

const HeavyChart = dynamic(() => import('@/components/HeavyChart'), {
  loading: () => <p>Loading chart...</p>,
  ssr: false, // Disable SSR if component uses browser APIs
});

// Route-based code splitting (automatic in Next.js app directory)
```

### Image Optimization

```typescript
import Image from 'next/image';

// Good: Use Next.js Image component
<Image
  src='/photo.jpg'
  alt='Description'
  width={500}
  height={300}
  priority={false} // Use priority={true} for above-fold images
  placeholder='blur'
  blurDataURL='...' // Low-quality placeholder
/>

// Avoid: Regular img tag for large images
<img src='/large-photo.jpg' alt='Description' />
```

## TypeScript Best Practices

### Strict Type Definitions

```typescript
// Good: Explicit types
interface User {
  id: number;
  name: string;
  email: string;
  role: 'admin' | 'user' | 'guest';
}

// Avoid: any type
const user: any = fetchUser();

// Good: Generic components
interface ListProps<T> {
  items: T[];
  renderItem: (item: T) => ReactNode;
}

export const List = <T,>({ items, renderItem }: ListProps<T>) => (
  <ul>{items.map(renderItem)}</ul>
);
```

### Utility Types

```typescript
// Use TypeScript utility types
type UserWithoutId = Omit<User, 'id'>;
type PartialUser = Partial<User>;
type ReadonlyUser = Readonly<User>;
type UserKeys = keyof User;

// Extract props from existing components
type ButtonProps = React.ComponentProps<typeof Button>;
```

## Error Handling

### Error Boundaries

```typescript
// Wrap components that might error
<ErrorBoundary fallback={<ErrorFallback />}>
  <ComponentThatMightError />
</ErrorBoundary>
```

### Async Error Handling

```typescript
// Good: Handle all error cases
const fetchData = async () => {
  try {
    const data = await apiClient.get('/data');
    return data;
  } catch (error) {
    if (error instanceof ApiError) {
      console.error('API Error:', error.message);
    } else {
      console.error('Unexpected error:', error);
    }
    throw error;
  }
};
```

## Code Organization

### File Structure

```
app/
├── (auth)/
│   ├── login/
│   └── register/
├── (dashboard)/
│   ├── profile/
│   └── settings/
├── api/
├── layout.tsx
└── page.tsx

components/
├── ui/              # Shared UI components
├── features/        # Feature-specific components
└── layouts/         # Layout components

hooks/               # Custom hooks
lib/                 # Utilities, API clients
types/               # TypeScript types
```

### Import Organization

```typescript
// Group and order imports
// 1. React and Next.js
import { useState, useEffect } from 'react';
import Image from 'next/image';

// 2. External libraries
import { useQuery } from '@tanstack/react-query';
import clsx from 'clsx';

// 3. Internal utilities and hooks
import { apiClient } from '@/lib/api-client';
import { useAuth } from '@/hooks/useAuth';

// 4. Components
import { Button } from '@/components/ui/Button';

// 5. Types
import type { User } from '@/types';

// 6. Styles (if needed)
import styles from './Component.module.css';
```

## Testing Considerations

### Component Testing

```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import { Button } from './Button';

describe('Button', () => {
  it('renders with correct text', () => {
    render(<Button label='Click me' />);
    expect(screen.getByText('Click me')).toBeInTheDocument();
  });

  it('calls onClick when clicked', () => {
    const handleClick = jest.fn();
    render(<Button label='Click me' onClick={handleClick} />);
    fireEvent.click(screen.getByText('Click me'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('is disabled when disabled prop is true', () => {
    render(<Button label='Click me' disabled />);
    expect(screen.getByText('Click me')).toBeDisabled();
  });
});
```

## Security Best Practices

### Sanitize User Input

```typescript
// Avoid direct HTML injection
const UserComment = ({ comment }: { comment: string }) => {
  return <div>{comment}</div>; // React escapes by default
};

// If you must use dangerouslySetInnerHTML, sanitize first
import DOMPurify from 'dompurify';

const UserComment = ({ htmlComment }: { htmlComment: string }) => {
  const sanitized = DOMPurify.sanitize(htmlComment);
  return <div dangerouslySetInnerHTML={{ __html: sanitized }} />;
};
```

### Environment Variables

```typescript
// Client-side variables must be prefixed with NEXT_PUBLIC_
const apiUrl = process.env.NEXT_PUBLIC_API_URL;

// Never expose secrets to client
// WRONG: const secret = process.env.SECRET_KEY; (in client component)
// RIGHT: Use secret only in server components or API routes
```

## Naming Conventions

### Files and Folders

- **Components**: PascalCase (`UserProfile.tsx`)
- **Utilities**: camelCase (`formatDate.ts`)
- **Hooks**: camelCase with 'use' prefix (`useAuth.ts`)
- **Types**: PascalCase (`UserTypes.ts` or `types.ts`)
- **Constants**: UPPER_SNAKE_CASE (`API_CONSTANTS.ts`)

### Variables and Functions

```typescript
// Boolean variables: use is/has/should prefix
const isLoading = true;
const hasError = false;
const shouldRender = true;

// Event handlers: use handle prefix
const handleClick = () => {};
const handleSubmit = () => {};

// Constants: UPPER_SNAKE_CASE
const MAX_RETRIES = 3;
const API_BASE_URL = 'https://api.example.com';
```

## Documentation

### Component Documentation

```typescript
/**
 * Button component with multiple variants and states
 * 
 * @example
 * ```tsx
 * <Button 
 *   label="Click me" 
 *   variant="primary" 
 *   onClick={() => console.log('clicked')}
 * />
 * ```
 */
interface ButtonProps {
  /** Text to display on the button */
  label: string;
  /** Click handler */
  onClick?: () => void;
  /** Visual style variant */
  variant?: 'primary' | 'secondary' | 'outline';
  /** Disabled state */
  disabled?: boolean;
}

export const Button = ({ label, onClick, variant = 'primary', disabled = false }: ButtonProps) => {
  // ...
};
```

## Common Anti-Patterns to Avoid

1. **Modifying state directly**: Always use setState
2. **Using index as key**: Use unique IDs instead
3. **Fetching in useEffect without cleanup**: Handle component unmounting
4. **Not handling loading/error states**: Always show feedback
5. **Ignoring accessibility**: Add ARIA labels and keyboard support
6. **Massive components**: Break into smaller pieces
7. **Prop drilling**: Use composition or context
8. **Premature optimization**: Optimize when needed, not by default
9. **Ignoring TypeScript errors**: Fix them, don't suppress
10. **Mixing concerns**: Separate logic, UI, and data fetching
