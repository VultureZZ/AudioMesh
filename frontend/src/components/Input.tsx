/**
 * Reusable input component
 */

import { InputHTMLAttributes, TextareaHTMLAttributes } from 'react';

interface BaseInputProps {
  label?: string;
  error?: string;
  helpText?: string;
}

interface TextInputProps extends InputHTMLAttributes<HTMLInputElement>, BaseInputProps {
  type?: 'text' | 'email' | 'password' | 'url';
  multiline?: false;
}

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement>, BaseInputProps {
  multiline: true;
}

type InputProps = TextInputProps | TextareaProps;

export function Input(props: InputProps) {
  const { label, error, helpText, className = '', ...restProps } = props;
  const isMultiline = 'multiline' in props && props.multiline;

  const baseInputClasses = 'w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500';
  const errorClasses = error ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : 'border-gray-300';

  // Important: Do NOT forward the `multiline` prop to the DOM (React warns).
  const { multiline: _multiline, ...inputProps } = restProps as Omit<InputProps, 'multiline'> & {
    multiline?: boolean;
  };

  const inputElement = isMultiline ? (
    <textarea
      className={`${baseInputClasses} ${errorClasses} ${className}`}
      {...(inputProps as TextareaHTMLAttributes<HTMLTextAreaElement>)}
    />
  ) : (
    <input
      className={`${baseInputClasses} ${errorClasses} ${className}`}
      {...(inputProps as InputHTMLAttributes<HTMLInputElement>)}
    />
  );

  return (
    <div className="w-full">
      {label && (
        <label className="block text-sm font-medium text-gray-700 mb-1">
          {label}
          {inputProps.required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}
      {inputElement}
      {error && <p className="mt-1 text-sm text-red-600">{error}</p>}
      {helpText && !error && <p className="mt-1 text-sm text-gray-500">{helpText}</p>}
    </div>
  );
}