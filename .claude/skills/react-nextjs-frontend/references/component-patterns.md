# Component Patterns

Detailed templates and patterns for React/Next.js components.

## Basic Component Template

### Client Component (Interactive)

```typescript
'use client';

import { useState } from 'react';

interface ButtonProps {
  label: string;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'outline';
  disabled?: boolean;
}

export const Button = ({ 
  label, 
  onClick, 
  variant = 'primary',
  disabled = false 
}: ButtonProps) => {
  const [isLoading, setIsLoading] = useState(false);

  const baseStyles = 'px-4 py-2 rounded-lg font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2';
  
  const variantStyles = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500',
    secondary: 'bg-gray-200 text-gray-900 hover:bg-gray-300 focus:ring-gray-500',
    outline: 'border-2 border-blue-600 text-blue-600 hover:bg-blue-50 focus:ring-blue-500'
  };

  const disabledStyles = 'opacity-50 cursor-not-allowed';

  return (
    <button
      onClick={onClick}
      disabled={disabled || isLoading}
      className={`${baseStyles} ${variantStyles[variant]} ${disabled ? disabledStyles : ''}`}
      aria-label={label}
    >
      {isLoading ? 'Loading...' : label}
    </button>
  );
};
```

### Server Component (Data Fetching)

```typescript
import { Suspense } from 'react';

interface Post {
  id: number;
  title: string;
  content: string;
}

async function getPosts(): Promise<Post[]> {
  const res = await fetch('https://api.example.com/posts', {
    cache: 'no-store' // or 'force-cache' for static data
  });
  
  if (!res.ok) throw new Error('Failed to fetch posts');
  return res.json();
}

export const PostList = async () => {
  const posts = await getPosts();

  return (
    <div className='space-y-4'>
      {posts.map((post) => (
        <article key={post.id} className='p-4 border rounded-lg'>
          <h2 className='text-xl font-semibold mb-2'>{post.title}</h2>
          <p className='text-gray-700'>{post.content}</p>
        </article>
      ))}
    </div>
  );
};

// Usage in page with loading state
export default function PostsPage() {
  return (
    <main className='container mx-auto p-4'>
      <h1 className='text-3xl font-bold mb-6'>Posts</h1>
      <Suspense fallback={<LoadingSkeleton />}>
        <PostList />
      </Suspense>
    </main>
  );
}
```

## Form Component Pattern

```typescript
'use client';

import { useState, FormEvent } from 'react';

interface FormData {
  name: string;
  email: string;
  message: string;
}

export const ContactForm = () => {
  const [formData, setFormData] = useState<FormData>({
    name: '',
    email: '',
    message: ''
  });
  const [errors, setErrors] = useState<Partial<FormData>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState<'idle' | 'success' | 'error'>('idle');

  const validateForm = (): boolean => {
    const newErrors: Partial<FormData> = {};

    if (!formData.name.trim()) {
      newErrors.name = 'Name is required';
    }

    if (!formData.email.trim()) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid';
    }

    if (!formData.message.trim()) {
      newErrors.message = 'Message is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    if (!validateForm()) return;

    setIsSubmitting(true);
    setSubmitStatus('idle');

    try {
      const response = await fetch('/api/contact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      if (!response.ok) throw new Error('Submission failed');

      setSubmitStatus('success');
      setFormData({ name: '', email: '', message: '' });
    } catch (error) {
      setSubmitStatus('error');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className='max-w-md mx-auto space-y-4'>
      <div>
        <label htmlFor='name' className='block text-sm font-medium mb-1'>
          Name
        </label>
        <input
          id='name'
          type='text'
          value={formData.name}
          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
          className='w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent'
          aria-invalid={!!errors.name}
          aria-describedby={errors.name ? 'name-error' : undefined}
        />
        {errors.name && (
          <p id='name-error' className='text-red-600 text-sm mt-1' role='alert'>
            {errors.name}
          </p>
        )}
      </div>

      <div>
        <label htmlFor='email' className='block text-sm font-medium mb-1'>
          Email
        </label>
        <input
          id='email'
          type='email'
          value={formData.email}
          onChange={(e) => setFormData({ ...formData, email: e.target.value })}
          className='w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent'
          aria-invalid={!!errors.email}
          aria-describedby={errors.email ? 'email-error' : undefined}
        />
        {errors.email && (
          <p id='email-error' className='text-red-600 text-sm mt-1' role='alert'>
            {errors.email}
          </p>
        )}
      </div>

      <div>
        <label htmlFor='message' className='block text-sm font-medium mb-1'>
          Message
        </label>
        <textarea
          id='message'
          value={formData.message}
          onChange={(e) => setFormData({ ...formData, message: e.target.value })}
          rows={4}
          className='w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent'
          aria-invalid={!!errors.message}
          aria-describedby={errors.message ? 'message-error' : undefined}
        />
        {errors.message && (
          <p id='message-error' className='text-red-600 text-sm mt-1' role='alert'>
            {errors.message}
          </p>
        )}
      </div>

      <button
        type='submit'
        disabled={isSubmitting}
        className='w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors'
      >
        {isSubmitting ? 'Sending...' : 'Send Message'}
      </button>

      {submitStatus === 'success' && (
        <p className='text-green-600 text-center' role='status'>
          Message sent successfully!
        </p>
      )}

      {submitStatus === 'error' && (
        <p className='text-red-600 text-center' role='alert'>
          Failed to send message. Please try again.
        </p>
      )}
    </form>
  );
};
```

## Custom Hook Pattern

```typescript
import { useState, useEffect } from 'react';

interface FetchState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
}

export const useFetch = <T,>(url: string) => {
  const [state, setState] = useState<FetchState<T>>({
    data: null,
    loading: true,
    error: null
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(url);
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        setState({ data, loading: false, error: null });
      } catch (error) {
        setState({ data: null, loading: false, error: error as Error });
      }
    };

    fetchData();
  }, [url]);

  return state;
};

// Usage
export const UserProfile = ({ userId }: { userId: string }) => {
  const { data, loading, error } = useFetch<User>(`/api/users/${userId}`);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  if (!data) return null;

  return (
    <div>
      <h1>{data.name}</h1>
      <p>{data.email}</p>
    </div>
  );
};
```

## Modal/Dialog Pattern

```typescript
'use client';

import { useEffect, useRef } from 'react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}

export const Modal = ({ isOpen, onClose, title, children }: ModalProps) => {
  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen) {
      // Focus trap
      modalRef.current?.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div 
      className='fixed inset-0 z-50 flex items-center justify-center p-4 bg-black bg-opacity-50'
      onClick={onClose}
      role='dialog'
      aria-modal='true'
      aria-labelledby='modal-title'
    >
      <div 
        ref={modalRef}
        className='bg-white rounded-lg shadow-xl max-w-md w-full p-6'
        onClick={(e) => e.stopPropagation()}
        tabIndex={-1}
      >
        <div className='flex items-center justify-between mb-4'>
          <h2 id='modal-title' className='text-xl font-semibold'>
            {title}
          </h2>
          <button
            onClick={onClose}
            className='text-gray-500 hover:text-gray-700'
            aria-label='Close modal'
          >
            <svg className='w-6 h-6' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
              <path strokeLinecap='round' strokeLinejoin='round' strokeWidth={2} d='M6 18L18 6M6 6l12 12' />
            </svg>
          </button>
        </div>
        <div>{children}</div>
      </div>
    </div>
  );
};
```

## Loading States

```typescript
// Skeleton Loader
export const SkeletonCard = () => (
  <div className='animate-pulse'>
    <div className='h-48 bg-gray-300 rounded-lg mb-4'></div>
    <div className='h-4 bg-gray-300 rounded w-3/4 mb-2'></div>
    <div className='h-4 bg-gray-300 rounded w-1/2'></div>
  </div>
);

// Spinner
export const Spinner = () => (
  <div className='flex justify-center items-center'>
    <div className='animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600'></div>
  </div>
);
```

## Error Boundary Pattern

```typescript
'use client';

import { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className='p-4 bg-red-50 border border-red-200 rounded-lg'>
          <h2 className='text-lg font-semibold text-red-800 mb-2'>Something went wrong</h2>
          <p className='text-red-600'>{this.state.error?.message}</p>
        </div>
      );
    }

    return this.props.children;
  }
}
```
