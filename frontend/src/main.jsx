import './index.css';

import React from 'react';

import { PostHogProvider } from 'posthog-js/react';
import ReactDOM from 'react-dom/client';
import {
  BrowserRouter,
  Route,
  Routes,
} from 'react-router-dom';

import Demo from './Demo';
import Landing from './Landing';

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
        </Routes>
      </BrowserRouter>
    </PostHogProvider>
  </React.StrictMode>
);
