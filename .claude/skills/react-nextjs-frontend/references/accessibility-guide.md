# Accessibility Guide

WCAG 2.1 AA compliance patterns and best practices for React/Next.js applications.

## Core Accessibility Principles

### 1. Perceivable

Users must be able to perceive the information being presented.

#### Color Contrast

Ensure sufficient color contrast ratios:
- **Normal text**: 4.5:1 minimum
- **Large text** (18pt+ or 14pt+ bold): 3:1 minimum
- **UI components and graphics**: 3:1 minimum

```tsx
// Good: High contrast
<button className='bg-blue-600 text-white'>Click Me</button>

// Bad: Low contrast
<button className='bg-gray-300 text-gray-400'>Click Me</button>
```

#### Alternative Text for Images

Always provide meaningful alt text:

```tsx
// Good: Descriptive alt text
<img src='/chart.png' alt='Sales revenue chart showing 20% increase in Q4 2024' />

// Decorative images
<img src='/decoration.png' alt='' role='presentation' />

// Using Next.js Image
import Image from 'next/image';

<Image
  src='/product.jpg'
  alt='Blue wireless headphones with noise cancellation'
  width={300}
  height={200}
/>
```

### 2. Operable

Users must be able to operate the interface.

#### Keyboard Navigation

All interactive elements must be keyboard accessible:

```tsx
export const DropdownMenu = () => {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const handleKeyDown = (e: KeyboardEvent) => {
    switch (e.key) {
      case 'Escape':
        setIsOpen(false);
        break;
      case 'ArrowDown':
        e.preventDefault();
        // Focus next item
        break;
      case 'ArrowUp':
        e.preventDefault();
        // Focus previous item
        break;
    }
  };

  return (
    <div ref={menuRef} onKeyDown={handleKeyDown}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        aria-haspopup='true'
      >
        Menu
      </button>
      {isOpen && (
        <ul role='menu'>
          <li role='menuitem' tabIndex={0}>Option 1</li>
          <li role='menuitem' tabIndex={0}>Option 2</li>
        </ul>
      )}
    </div>
  );
};
```

#### Focus Management

Provide visible focus indicators and manage focus appropriately:

```css
/* In global CSS or Tailwind config */
*:focus-visible {
  outline: 2px solid #3b82f6;
  outline-offset: 2px;
}

/* Or with Tailwind */
className='focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2'
```

```tsx
// Focus management in modals
export const Modal = ({ isOpen, onClose }: ModalProps) => {
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (isOpen) {
      closeButtonRef.current?.focus();
    }
  }, [isOpen]);

  return (
    <dialog open={isOpen} aria-modal='true'>
      <button ref={closeButtonRef} onClick={onClose}>
        Close
      </button>
    </dialog>
  );
};
```

#### Skip Links

Allow users to skip repetitive navigation:

```tsx
export const SkipLink = () => (
  <a
    href='#main-content'
    className='sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:bg-blue-600 focus:text-white focus:px-4 focus:py-2 focus:rounded'
  >
    Skip to main content
  </a>
);

// In layout
export default function Layout({ children }: { children: ReactNode }) {
  return (
    <>
      <SkipLink />
      <nav>...</nav>
      <main id='main-content'>{children}</main>
    </>
  );
}
```

### 3. Understandable

Users must be able to understand the information and interface.

#### Form Labels and Instructions

Always associate labels with form controls:

```tsx
// Good: Proper label association
<div>
  <label htmlFor='email'>Email Address</label>
  <input
    id='email'
    type='email'
    aria-required='true'
    aria-describedby='email-hint'
  />
  <p id='email-hint' className='text-sm text-gray-600'>
    We'll never share your email with anyone else.
  </p>
</div>

// Error states
<div>
  <label htmlFor='password'>Password</label>
  <input
    id='password'
    type='password'
    aria-invalid={hasError}
    aria-describedby='password-error'
  />
  {hasError && (
    <p id='password-error' className='text-red-600' role='alert'>
      Password must be at least 8 characters.
    </p>
  )}
</div>
```

#### Clear Error Messages

Provide specific, actionable error messages:

```tsx
// Good: Specific error message
<p role='alert'>
  Password must contain at least one uppercase letter, one number, and be 8+ characters long.
</p>

// Bad: Vague error message
<p role='alert'>Invalid password</p>
```

### 4. Robust

Content must be robust enough to work with various user agents and assistive technologies.

#### Semantic HTML

Use appropriate HTML elements:

```tsx
// Good: Semantic HTML
<nav aria-label='Main navigation'>
  <ul>
    <li><a href='/'>Home</a></li>
    <li><a href='/about'>About</a></li>
  </ul>
</nav>

<article>
  <h1>Article Title</h1>
  <p>Content...</p>
</article>

// Bad: Div soup
<div className='nav'>
  <div className='nav-item'>Home</div>
  <div className='nav-item'>About</div>
</div>
```

#### ARIA Attributes

Use ARIA attributes appropriately:

```tsx
// Button that toggles content
<button
  aria-expanded={isOpen}
  aria-controls='panel-content'
  onClick={() => setIsOpen(!isOpen)}
>
  Toggle Panel
</button>

<div id='panel-content' hidden={!isOpen}>
  Panel content
</div>

// Loading state
<div role='status' aria-live='polite'>
  {isLoading ? 'Loading...' : 'Content loaded'}
</div>

// Icon button
<button aria-label='Delete item'>
  <TrashIcon />
</button>
```

## Common Accessible Components

### Accessible Button

```tsx
interface ButtonProps {
  children: ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary';
  disabled?: boolean;
  ariaLabel?: string;
  type?: 'button' | 'submit' | 'reset';
}

export const AccessibleButton = ({
  children,
  onClick,
  variant = 'primary',
  disabled = false,
  ariaLabel,
  type = 'button'
}: ButtonProps) => (
  <button
    type={type}
    onClick={onClick}
    disabled={disabled}
    aria-label={ariaLabel}
    className={`px-4 py-2 rounded focus:outline-none focus:ring-2 focus:ring-offset-2 ${
      variant === 'primary' 
        ? 'bg-blue-600 text-white focus:ring-blue-500' 
        : 'bg-gray-200 text-gray-900 focus:ring-gray-500'
    } disabled:opacity-50 disabled:cursor-not-allowed`}
  >
    {children}
  </button>
);
```

### Accessible Tabs

```tsx
export const Tabs = ({ tabs }: { tabs: Array<{ id: string; label: string; content: ReactNode }> }) => {
  const [activeTab, setActiveTab] = useState(tabs[0].id);

  return (
    <div>
      <div role='tablist' aria-label='Content tabs'>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            role='tab'
            aria-selected={activeTab === tab.id}
            aria-controls={`panel-${tab.id}`}
            id={`tab-${tab.id}`}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 ${activeTab === tab.id ? 'border-b-2 border-blue-600' : ''}`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {tabs.map((tab) => (
        <div
          key={tab.id}
          role='tabpanel'
          id={`panel-${tab.id}`}
          aria-labelledby={`tab-${tab.id}`}
          hidden={activeTab !== tab.id}
        >
          {tab.content}
        </div>
      ))}
    </div>
  );
};
```

### Accessible Tooltip

```tsx
export const Tooltip = ({ children, content }: { children: ReactNode; content: string }) => {
  const [isVisible, setIsVisible] = useState(false);
  const tooltipId = useId();

  return (
    <div className='relative inline-block'>
      <div
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onFocus={() => setIsVisible(true)}
        onBlur={() => setIsVisible(false)}
        aria-describedby={tooltipId}
      >
        {children}
      </div>
      {isVisible && (
        <div
          id={tooltipId}
          role='tooltip'
          className='absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-sm rounded whitespace-nowrap'
        >
          {content}
        </div>
      )}
    </div>
  );
};
```

## Screen Reader Only Content

Use the `sr-only` utility for screen reader only content:

```tsx
// Tailwind sr-only utility
<span className='sr-only'>Additional context for screen readers</span>

// Custom CSS
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}

.sr-only:focus {
  position: static;
  width: auto;
  height: auto;
  padding: inherit;
  margin: inherit;
  overflow: visible;
  clip: auto;
  white-space: normal;
}
```

## Testing Accessibility

### Tools

1. **Automated Testing**: axe DevTools, Lighthouse, WAVE
2. **Manual Testing**: Keyboard navigation, screen reader testing
3. **Libraries**: @testing-library/react with accessibility matchers

### Keyboard Testing Checklist

- [ ] Tab through all interactive elements
- [ ] Shift+Tab to navigate backwards
- [ ] Enter/Space to activate buttons and links
- [ ] Arrow keys for custom controls (menus, tabs)
- [ ] Escape to close modals and menus
- [ ] Focus indicators are visible and clear

### Screen Reader Testing

Test with at least one screen reader:
- **Windows**: NVDA (free), JAWS
- **macOS**: VoiceOver (built-in)
- **Linux**: Orca

## Common Accessibility Issues to Avoid

1. **Missing alt text** on images
2. **Low color contrast** between text and background
3. **Missing form labels** or improper label association
4. **Keyboard traps** where users can't escape from a component
5. **Missing focus indicators** on interactive elements
6. **Using divs/spans** for buttons or links
7. **Auto-playing media** without controls
8. **Time-limited content** without option to extend
9. **Missing skip links** for navigation
10. **Improper heading hierarchy** (skipping levels)
