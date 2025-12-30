import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  IdleScreen,
  LoadingSteps,
  PromptBanner,
  TranscriptBubble,
  ThinkingSection,
  ResponseSection,
  BottomControls,
} from './components/voice';
import { useWebSocketVoice } from './hooks/useWebSocketVoice';

const SAMPLE_STORIES = [
  {
    title: 'The Lighthouse Keeper',
    text: `There was once a lighthouse keeper named Thomas who lived alone on a rocky island. Every night, he climbed the spiral stairs to light the lamp. One stormy evening, he saw a ship heading straight for the rocks. He waved his lantern frantically, but the ship didn't turn. At the last moment, he remembered the old foghorn in the basement. He ran down and cranked it to life. The deep sound cut through the storm, and the ship finally turned away, just missing the rocks.`,
  },
  {
    title: 'The Last Library',
    text: `In the year 2157, books had become extinct. Everything was digital, streamed directly to neural implants. But in a forgotten basement of old New York, a woman named Maya discovered something impossible: a room full of paper books. She picked one up and felt its weight, smelled its pages. As she read the first sentence aloud, she understood why they had been hidden. The books contained truths that couldn't be deleted.`,
  },
  {
    title: "The Clockmaker's Gift",
    text: `Old Mr. Chen was the finest clockmaker in the city, but he never sold his most beautiful clock. It sat in his window, its gears visible through crystal glass, ticking perfectly. A young girl asked him why. He smiled and said, "This clock keeps special time. It only moves forward when something kind happens in the world." The girl watched it tick. Then she ran outside and helped an elderly woman cross the street. When she looked back through the window, she saw the clock had moved three seconds.`,
  },
];

export default function VoiceWebSocket() {
  const [showPrompt, setShowPrompt] = useState(true);
  const [selectedStory] = useState(() => Math.floor(Math.random() * SAMPLE_STORIES.length));
  const chatRef = useRef(null);

  // Server URL - Modal endpoint (class-based deployment)
  const serverUrl = 'https://bryanhoulton--websocket-voice-agent-voiceagent-web-app.modal.run';

  const {
    stage,
    connect,
    disconnect,
    error,
    setError,
    isRecording,
    toggleRecording,
    displayedTranscript,
    userSpeaking,
    currentThinking,
    isThinking,
    currentResponse,
    isSpeaking,
    responseLatency,
    eagerEndOfTurn,
    modelContext,
    thinkingStalled,
    stallInfo,
  } = useWebSocketVoice(serverUrl);
  
  const [showContext, setShowContext] = useState(false);

  // Auto-scroll chat area
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [displayedTranscript, currentResponse]);

  const hasContent = displayedTranscript || currentThinking || currentResponse;
  const story = SAMPLE_STORIES[selectedStory];

  // Idle state
  if (stage === 'idle') {
    return (
      <IdleScreen
        onStart={() => {
          connect();
        }}
        error={error}
      />
    );
  }

  // Loading state (connecting)
  if (stage === 'connecting') {
    return <LoadingSteps loadingStep="connecting" />;
  }

  // Error state
  if (stage === 'error') {
    return (
      <IdleScreen
        onStart={() => {
          setError(null);
          connect();
        }}
        error={error}
      />
    );
  }

  // Active state (connected)
  return (
    <div className="h-screen overflow-hidden bg-white flex flex-col">
      {/* Top bar */}
      <div className="flex-shrink-0 px-4 py-3 flex justify-between items-center border-b border-[#eee]">
        <Link
          to="/"
          onClick={disconnect}
          className="text-xs text-[#999] hover:text-[#666] flex items-center gap-1 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back
        </Link>

        {/* Status indicators */}
        <div className="flex gap-2">
          <AnimatePresence>
            {thinkingStalled && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="px-2 py-1 bg-amber-100 text-amber-700 text-xs rounded-full font-medium"
                title={stallInfo ? JSON.stringify(stallInfo) : ''}
              >
                ⚠️ Thinking Stalled
              </motion.div>
            )}
          </AnimatePresence>
          <AnimatePresence>
            {eagerEndOfTurn && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full font-medium"
              >
                🚀 EagerEndOfTurn
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => setShowContext(!showContext)}
            className={`text-xs px-2 py-1 rounded ${showContext ? 'bg-blue-100 text-blue-700' : 'text-[#999] hover:text-[#666]'} transition-colors`}
          >
            {showContext ? 'Hide' : 'Show'} context
          </button>
          <button
            onClick={() => setShowPrompt(!showPrompt)}
            className="text-xs text-[#999] hover:text-[#666] flex items-center gap-1 transition-colors"
          >
            {showPrompt ? 'Hide' : 'Show'} prompt
            <svg
              className={`w-3 h-3 transition-transform ${showPrompt ? 'rotate-180' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
      </div>

      {/* Model context panel */}
      <AnimatePresence>
        {showContext && modelContext && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-b border-[#eee] bg-[#1a1a2e] overflow-hidden"
          >
            <pre className="p-4 text-xs text-green-400 font-mono overflow-x-auto max-h-64 overflow-y-auto whitespace-pre-wrap">
              {modelContext}
            </pre>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Prompt banner */}
      <PromptBanner show={showPrompt} title={story.title} text={story.text} />

      {/* Chat area */}
      <div ref={chatRef} className="flex-1 overflow-y-auto px-4">
        <div className="max-w-2xl mx-auto py-6 space-y-4">
          <TranscriptBubble
            displayedTranscript={displayedTranscript}
            liveTranscript={displayedTranscript}
            userSpeaking={userSpeaking}
          />

          <ThinkingSection thinking={currentThinking} isThinking={isThinking} />

          <ResponseSection
            response={currentResponse}
            latency={responseLatency}
            isSpeaking={isSpeaking}
          />

          {/* Empty state */}
          {!hasContent && (
            <div className="text-center py-12">
              <p className="text-[#bbb]">Start speaking to see the AI think in real-time</p>
            </div>
          )}
        </div>
      </div>

      {/* Bottom controls */}
      <BottomControls
        isRecording={isRecording}
        onToggleRecording={toggleRecording}
        userSpeaking={userSpeaking}
        isThinking={isThinking}
        hasThinking={!!currentThinking}
        hasResponse={!!currentResponse}
        isSpeaking={isSpeaking}
      />

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="absolute bottom-24 left-1/2 -translate-x-1/2 bg-red-50 text-red-600 px-4 py-2 rounded-lg text-sm"
          >
            {error}
            <button onClick={() => setError(null)} className="ml-2 text-red-400 hover:text-red-600">
              ×
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

