/**
 * Alert/toast component for success/error messages
 */

import { useEffect, useState } from 'react';

interface AlertProps {
  type?: 'success' | 'error' | 'info' | 'warning';
  message: string;
  onClose?: () => void;
  autoDismiss?: boolean;
  dismissDelay?: number;
}

export function Alert({ type = 'info', message, onClose, autoDismiss = false, dismissDelay = 5000 }: AlertProps) {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    if (autoDismiss && isVisible) {
      const timer = setTimeout(() => {
        setIsVisible(false);
        onClose?.();
      }, dismissDelay);
      return () => clearTimeout(timer);
    }
  }, [autoDismiss, dismissDelay, isVisible, onClose]);

  if (!isVisible) return null;

  const typeClasses = {
    success: 'bg-green-50 text-green-800 border-green-200',
    error: 'bg-red-50 text-red-800 border-red-200',
    info: 'bg-blue-50 text-blue-800 border-blue-200',
    warning: 'bg-yellow-50 text-yellow-800 border-yellow-200',
  };

  const icon = {
    success: '✓',
    error: '✕',
    info: 'ℹ',
    warning: '⚠',
  };

  return (
    <div className={`border rounded-md p-4 ${typeClasses[type]} flex items-start gap-3`}>
      <span className="font-semibold">{icon[type]}</span>
      <div className="flex-1">{message}</div>
      {onClose && (
        <button onClick={() => { setIsVisible(false); onClose(); }} className="hover:opacity-70">
          ×
        </button>
      )}
    </div>
  );
}

/**
 * Toast notification container
 */
interface Toast {
  id: string;
  type: 'success' | 'error' | 'info' | 'warning';
  message: string;
}

interface ToastContainerProps {
  toasts: Toast[];
  onRemove: (id: string) => void;
}

export function ToastContainer({ toasts, onRemove }: ToastContainerProps) {
  return (
    <div className="fixed top-4 right-4 z-50 space-y-2">
      {toasts.map((toast) => (
        <Alert
          key={toast.id}
          type={toast.type}
          message={toast.message}
          autoDismiss
          onClose={() => onRemove(toast.id)}
        />
      ))}
    </div>
  );
}