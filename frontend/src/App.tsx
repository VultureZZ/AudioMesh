import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/Layout';
import { GeneratePage } from './pages/GeneratePage';
import { RealtimeGeneratePage } from './pages/RealtimeGeneratePage';
import { PodcastPage } from './pages/PodcastPage';
import { PodcastsLibraryPage } from './pages/PodcastsLibraryPage';
import { MusicPage } from './pages/MusicPage';
import { VoicesPage } from './pages/VoicesPage';
import { SettingsPage } from './pages/SettingsPage';
import { TranscriptsPage } from './pages/TranscriptsPage';
import { AdScannerPage } from './pages/AudioTools/AdScanner';
import { VoiceIsolatorPage } from './pages/AudioTools/VoiceIsolator';

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/generate" replace />} />
          <Route path="/generate" element={<GeneratePage />} />
          <Route path="/realtime" element={<RealtimeGeneratePage />} />
          <Route path="/podcast" element={<PodcastPage />} />
          <Route path="/music" element={<MusicPage />} />
          <Route path="/podcasts" element={<PodcastsLibraryPage />} />
          <Route path="/voices" element={<VoicesPage />} />
          <Route path="/transcripts" element={<TranscriptsPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/audio-tools" element={<Navigate to="/audio-tools/ad-scanner" replace />} />
          <Route path="/audio-tools/ad-scanner" element={<AdScannerPage />} />
          <Route path="/audio-tools/voice-isolator" element={<VoiceIsolatorPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;