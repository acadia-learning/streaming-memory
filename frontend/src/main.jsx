import './index.css';

import React from 'react';

import { PostHogProvider } from 'posthog-js/react';
import ReactDOM from 'react-dom/client';
import {
  BrowserRouter,
  Route,
  Routes,
} from 'react-router-dom';

import AryanDemo from './AryanDemo';
import Demo from './Demo';
import Landing from './Landing';
import VoiceDemo from './VoiceDemo';
import VoiceLiveKit from './VoiceLiveKit';
import VoiceWebSocket from './VoiceWebSocket';

const options = {
  api_host: import.meta.env.VITE_PUBLIC_POSTHOG_HOST,
};

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <PostHogProvider
      apiKey={import.meta.env.VITE_PUBLIC_POSTHOG_KEY}
      options={options}
    >
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/demo" element={<Demo />} />
          <Route path="/aryan" element={<AryanDemo />} />
          <Route path="/voice" element={<VoiceDemo />} />
          <Route path="/voice-livekit" element={<VoiceLiveKit />} />
          <Route path="/voice-ws" element={<VoiceWebSocket />} />
        </Routes>
      </BrowserRouter>
    </PostHogProvider>
  </React.StrictMode>
);
