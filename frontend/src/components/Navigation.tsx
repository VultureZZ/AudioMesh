/**
 * Navigation bar component
 */

import { Link, useLocation } from 'react-router-dom';
import { useState } from 'react';

const audioToolsChildren = [
  { path: '/audio-tools/ad-scanner', label: 'Ad Scanner' },
  { path: '/audio-tools/voice-isolator', label: 'Voice Isolator' },
];

export function Navigation() {
  const location = useLocation();
  const [audioToolsOpen, setAudioToolsOpen] = useState(false);

  const navLinks: { path: string; label: string }[] = [
    { path: '/generate', label: 'Generate' },
    { path: '/realtime', label: 'Realtime' },
    { path: '/podcast', label: 'Article Podcaster' },
    { path: '/music', label: 'Music' },
    { path: '/podcasts', label: 'Podcast Library' },
    { path: '/voices', label: 'Voices' },
    { path: '/transcripts', label: 'Transcripts' },
    { path: '/settings', label: 'Settings' },
  ];

  const isActive = (path: string) => location.pathname === path;
  const audioToolsActive = location.pathname.startsWith('/audio-tools');

  return (
    <nav className="bg-white border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <Link to="/" className="flex items-center">
              <span className="text-xl font-bold text-primary-600">AudioMesh</span>
            </Link>
            <div className="hidden sm:ml-6 sm:flex sm:items-center sm:space-x-6">
              {navLinks.map((link) => (
                <Link
                  key={link.path}
                  to={link.path}
                  className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium ${
                    isActive(link.path)
                      ? 'border-primary-500 text-gray-900'
                      : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                  }`}
                >
                  {link.label}
                </Link>
              ))}
              <div
                className="relative"
                onMouseEnter={() => setAudioToolsOpen(true)}
                onMouseLeave={() => setAudioToolsOpen(false)}
              >
                <button
                  type="button"
                  className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium ${
                    audioToolsActive
                      ? 'border-primary-500 text-gray-900'
                      : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                  }`}
                  aria-expanded={audioToolsOpen}
                  aria-haspopup="true"
                >
                  Audio Tools
                  <span className="ml-1 text-gray-400" aria-hidden>
                    ▾
                  </span>
                </button>
                {audioToolsOpen && (
                  <div className="absolute left-0 top-full pt-1 z-20 min-w-[12rem]">
                    <div className="rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 py-1">
                      {audioToolsChildren.map((child) => (
                        <Link
                          key={child.path}
                          to={child.path}
                          className={`block px-4 py-2 text-sm ${
                            isActive(child.path)
                              ? 'bg-primary-50 text-primary-800'
                              : 'text-gray-700 hover:bg-gray-50'
                          }`}
                        >
                          {child.label}
                        </Link>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      <div className="sm:hidden">
        <div className="pt-2 pb-3 space-y-1">
          {navLinks.map((link) => (
            <Link
              key={link.path}
              to={link.path}
              className={`block pl-3 pr-4 py-2 border-l-4 text-base font-medium ${
                isActive(link.path)
                  ? 'bg-primary-50 border-primary-500 text-primary-700'
                  : 'border-transparent text-gray-500 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-700'
              }`}
            >
              {link.label}
            </Link>
          ))}
          <div className="pl-3 pr-4 py-2 border-l-4 border-transparent text-gray-700 text-base font-semibold">
            Audio Tools
          </div>
          {audioToolsChildren.map((child) => (
            <Link
              key={child.path}
              to={child.path}
              className={`block pl-8 pr-4 py-2 border-l-4 text-base font-medium ${
                isActive(child.path)
                  ? 'bg-primary-50 border-primary-500 text-primary-700'
                  : 'border-transparent text-gray-500 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-700'
              }`}
            >
              {child.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
