import { useCallback, useEffect, useRef, useState } from 'react';

import { AnimatePresence, motion } from 'framer-motion';
import { Link } from 'react-router-dom';

const WS_URL =
  import.meta.env.VITE_VOICE_WS_URL ||
  'wss://bryanhoulton--voice-agent-voiceagent-serve.modal.run/ws';

export default function VoiceDemo() {
  const [isConnected, setIsConnected] = useState(false);
  const [isReady, setIsReady] = useState(false);  // DeepGram connected and ready
  const [isLoading, setIsLoading] = useState(false);  // Connecting/initializing
  const [isRecording, setIsRecording] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState('');
  const [currentThinking, setCurrentThinking] = useState('');
  const [currentResponse, setCurrentResponse] = useState('');
  const [isThinking, setIsThinking] = useState(true);
  const [injectionCount, setInjectionCount] = useState(0);
  const [error, setError] = useState(null);
  const [audioLevel, setAudioLevel] = useState(0);
  const [status, setStatus] = useState('Click to start');
  const [statusStage, setStatusStage] = useState('idle');
  const [tokenCount, setTokenCount] = useState(0);

  const wsRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const thinkingRef = useRef(null);

  // Auto-scroll thinking
  useEffect(() => {
    if (thinkingRef.current) {
      thinkingRef.current.scrollTop = thinkingRef.current.scrollHeight;
    }
  }, [currentThinking]);

  // Handle server events
  const handleServerEvent = useCallback((data) => {
    switch (data.type) {
      case 'connected':
        setStatus('Connected');
        setStatusStage('connected');
        break;

      case 'status':
        setStatus(data.message);
        setStatusStage(data.stage);
        // DeepGram is ready - we can start recording
        if (data.stage === 'ready') {
          setIsReady(true);
          setIsLoading(false);
        }
        break;

      case 'transcript':
        // Update transcript immediately with every word
        setLiveTranscript(data.full_transcript || data.text);
        if (!data.is_final) {
          setStatus('Listening...');
          setStatusStage('listening');
        }
        break;

      case 'speech_started':
        setStatus('Listening...');
        setStatusStage('listening');
        break;

      case 'utterance_end':
        setStatus('Processing...');
        setStatusStage('processing');
        break;

      case 'generation_start':
        setCurrentThinking('');
        setCurrentResponse('');
        setTokenCount(0);
        setIsThinking(true);
        break;

      case 'context_injection':
        // Transcript injected mid-generation - DON'T clear!
        setInjectionCount((prev) => prev + 1);
        break;

      case 'thinking_start':
        setIsThinking(true);
        break;

      case 'thinking_end':
        setIsThinking(false);
        break;

      case 'response_complete':
        setStatus('Response complete');
        setStatusStage('complete');
        break;

      case 'token':
        if (data.is_thinking) {
          setCurrentThinking((prev) => prev + data.t);
        } else {
          setCurrentResponse((prev) => prev + data.t);
        }
        if (data.token_count) {
          setTokenCount(data.token_count);
        }
        break;

      case 'cleared':
        setLiveTranscript('');
        setCurrentThinking('');
        setCurrentResponse('');
        setInjectionCount(0);
        setTokenCount(0);
        setIsThinking(true);
        break;

      case 'error':
        setError(data.message);
        setStatus('Error');
        setStatusStage('error');
        break;

      default:
        break;
    }
  }, []);

  // Connect to WebSocket
  const connect = useCallback(async () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      setError(null);
      setIsLoading(true);
      setStatus('Connecting...');
      setStatusStage('connecting');
      
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        setIsConnected(true);
        setStatus('Initializing...');
        setStatusStage('initializing');
      };

      ws.onclose = () => {
        setIsConnected(false);
        setIsRecording(false);
        setIsReady(false);
        setIsLoading(false);
        setStatus('Disconnected');
        setStatusStage('disconnected');
      };

      ws.onerror = () => {
        setError('Connection failed');
        setStatus('Error');
        setStatusStage('error');
        setIsLoading(false);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleServerEvent(data);
        } catch (e) {
          console.error('Parse error:', e);
        }
      };

      wsRef.current = ws;
    } catch (e) {
      setError(`Connection failed: ${e.message}`);
      setStatus('Error');
      setStatusStage('error');
      setIsLoading(false);
    }
  }, [handleServerEvent]);

  // Ref to track if we should auto-start recording when ready
  const pendingStartRef = useRef(false);

  // Initialize - connect and wait for ready
  const initialize = useCallback(async () => {
    if (isLoading) return;
    
    setIsLoading(true);
    pendingStartRef.current = true;
    await connect();
    // Recording will start automatically when we receive 'ready' status
  }, [connect, isLoading]);

  // Start recording (internal)
  const doStartRecording = useCallback(async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    try {
      setStatus('Mic access...');
      setStatusStage('mic_request');
      
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      mediaStreamRef.current = stream;

      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;

      const updateAudioLevel = () => {
        if (!analyserRef.current) return;
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        setAudioLevel(average / 255);
        animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
      };
      updateAudioLevel();

      // Use smaller buffer (512 samples = 32ms at 16kHz) for faster transcription
      const processor = audioContext.createScriptProcessor(512, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (wsRef.current?.readyState !== WebSocket.OPEN) return;

        const inputData = e.inputBuffer.getChannelData(0);
        const int16Data = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        wsRef.current.send(int16Data.buffer);
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      setIsRecording(true);
      setStatus('Listening');
      setStatusStage('recording');
      
      // Tell backend to resume generation
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'resume' }));
      }
    } catch (e) {
      setError(`Mic error: ${e.message}`);
      setStatus('Mic error');
      setStatusStage('error');
      setIsLoading(false);
    }
  }, []);

  // Auto-start recording when ready (after initialization)
  useEffect(() => {
    if (isReady && pendingStartRef.current && !isRecording) {
      pendingStartRef.current = false;
      doStartRecording();
    }
  }, [isReady, isRecording, doStartRecording]);

  // Public startRecording - either init or resume
  const startRecording = useCallback(async () => {
    if (!isConnected || !isReady) {
      await initialize();
    } else {
      await doStartRecording();
    }
  }, [isConnected, isReady, initialize, doStartRecording]);

  // Stop recording
  const stopRecording = useCallback(() => {
    // Tell backend to pause generation
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'pause' }));
    }
    
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    setIsRecording(false);
    setAudioLevel(0);
    setStatus('Stopped');
    setStatusStage('stopped');
  }, []);

  // Clear everything
  const clearAll = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'clear' }));
    }
    setLiveTranscript('');
    setCurrentThinking('');
    setCurrentResponse('');
    setInjectionCount(0);
    setTokenCount(0);
    setIsThinking(true);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording();
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [stopRecording]);

  const hasContent = liveTranscript || currentThinking || currentResponse;

  // Empty state - centered
  if (!hasContent && !isRecording) {
    return (
      <div className="h-screen overflow-hidden bg-white flex flex-col items-center justify-center px-4 relative">
        {/* Top bar */}
        <div className="absolute top-4 left-4 right-4 flex justify-between items-center">
          <Link
            to="/"
            className="text-xs text-[#999] hover:text-[#666] flex items-center gap-1 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back
          </Link>

          <div className="flex items-center gap-2 text-xs text-[#999]">
            <div className={`w-2 h-2 rounded-full transition-colors ${
              statusStage === 'ready' ? 'bg-green-500 animate-pulse' :
              statusStage === 'deepgram_connecting' ? 'bg-yellow-500 animate-pulse' :
              isConnected ? 'bg-green-500' : 'bg-[#ccc]'
            }`} />
            <span className={statusStage === 'ready' ? 'text-green-600 font-medium' : ''}>{status}</span>
          </div>
        </div>

        <h1 className="text-2xl font-bold text-[#1a1a1a] mb-2">Voice Demo</h1>
        <p className="text-[#999] text-sm mb-8 max-w-md text-center">
          Speak and watch the LLM reason in real-time. As you talk, the model continuously adapts its thinking.
        </p>

        {/* Mic button or loading spinner */}
        {isLoading ? (
          <div className="relative p-6">
            {/* Spinning loader */}
            <svg className="w-12 h-12 animate-spin text-[#1a1a1a]" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          </div>
        ) : (
          <button
            onClick={startRecording}
            className="relative p-6 rounded-full bg-[#1a1a1a] hover:bg-[#333] transition-colors shadow-lg"
          >
            <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
              />
            </svg>
          </button>
        )}

        <p className="text-[#999] text-xs mt-4">
          {isLoading ? status : 'Click to start'}
        </p>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              className="absolute bottom-8 bg-red-50 text-red-600 px-4 py-2 rounded-lg text-sm"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  }

  // Active state - split view
  return (
    <div className="h-screen overflow-hidden bg-white flex flex-col">
      {/* Top bar */}
      <div className="flex-shrink-0 px-4 pt-3 flex justify-between items-center">
        <Link
          to="/"
          className="text-xs text-[#999] hover:text-[#666] flex items-center gap-1 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back
        </Link>

        <div className="flex items-center gap-4 text-xs">
          {/* Status */}
          <div className="flex items-center gap-2 text-[#999]">
            <div className={`w-2 h-2 rounded-full transition-colors ${
              statusStage === 'ready' ? 'bg-green-500 animate-pulse' :
              statusStage === 'listening' ? 'bg-green-500 animate-pulse' :
              statusStage === 'processing' ? 'bg-blue-500 animate-pulse' :
              statusStage === 'recording' ? 'bg-yellow-500' :
              statusStage === 'deepgram_connecting' ? 'bg-yellow-500 animate-pulse' :
              isConnected ? 'bg-green-500' : 'bg-[#ccc]'
            }`} />
            <span className={statusStage === 'ready' ? 'text-green-600 font-medium' : ''}>{status}</span>
          </div>
          {tokenCount > 0 && (
            <span className="text-[#bbb]">{tokenCount} tokens</span>
          )}
          {injectionCount > 0 && (
            <span className="text-[#bbb]">{injectionCount} injections</span>
          )}
          {!isThinking && currentResponse && (
            <span className="text-green-600">responding</span>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-hidden flex">
        {/* Left: Transcript */}
        <div className="w-1/2 border-r border-[#eee] flex flex-col">
          <div className="px-4 py-2 border-b border-[#eee]">
            <span className="text-xs text-[#999] uppercase tracking-wide">Your Speech</span>
          </div>
          
          <div className="flex-1 overflow-y-auto p-4">
            {liveTranscript ? (
              <p className="text-[#1a1a1a] leading-relaxed">
                {liveTranscript}
                <span className="inline-block w-0.5 h-4 bg-[#1a1a1a] ml-1 animate-pulse" />
              </p>
            ) : (
              <p className="text-[#ccc] italic">Waiting for speech...</p>
            )}
          </div>
        </div>

        {/* Right: Thinking + Response */}
        <div className="w-1/2 flex flex-col">
          <div className="px-4 py-2 border-b border-[#eee]">
            <span className="text-xs text-[#999] uppercase tracking-wide">
              {isThinking ? 'LLM Reasoning' : 'LLM Response'}
            </span>
          </div>
          
          <div ref={thinkingRef} className="flex-1 overflow-y-auto p-4">
            {currentThinking && (
              <div className="mb-4">
                <p className="text-[#999] leading-relaxed italic whitespace-pre-wrap text-sm">
                  {currentThinking}
                </p>
              </div>
            )}
            {currentResponse && (
              <div className="border-t border-[#eee] pt-4">
                <p className="text-[#1a1a1a] leading-relaxed whitespace-pre-wrap">
                  {currentResponse}
                </p>
              </div>
            )}
            {!currentThinking && !currentResponse && (
              <p className="text-[#ccc] italic">Waiting for reasoning...</p>
            )}
          </div>
        </div>
      </div>

      {/* Bottom controls */}
      <div className="flex-shrink-0 px-4 py-4 border-t border-[#eee] bg-[#fafafa]">
        <div className="flex items-center justify-center gap-4">
          {/* Clear button */}
          <button
            onClick={clearAll}
            className="p-2 rounded-full text-[#999] hover:text-[#666] hover:bg-[#eee] transition-colors"
            title="Clear"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
              />
            </svg>
          </button>

          {/* Mic button */}
          <button
            onClick={isRecording ? stopRecording : startRecording}
            className={`relative p-4 rounded-full transition-colors ${
              isRecording
                ? 'bg-red-500 hover:bg-red-600'
                : 'bg-[#1a1a1a] hover:bg-[#333]'
            }`}
          >
            {/* Audio level ring */}
            {isRecording && (
              <div
                className="absolute inset-0 rounded-full border-2 border-red-300 transition-transform"
                style={{
                  transform: `scale(${1 + audioLevel * 0.3})`,
                  opacity: 0.5 + audioLevel * 0.5,
                }}
              />
            )}

            <svg className="w-6 h-6 text-white relative z-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              {isRecording ? (
                <rect x="6" y="6" width="12" height="12" rx="2" strokeWidth={2} />
              ) : (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                />
              )}
            </svg>
          </button>

          {/* Placeholder for symmetry */}
          <div className="w-9" />
        </div>
      </div>

      {/* Error banner */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="absolute bottom-24 left-1/2 -translate-x-1/2 bg-red-50 text-red-600 px-4 py-2 rounded-lg text-sm shadow-lg"
          >
            {error}
            <button onClick={() => setError(null)} className="ml-2 text-red-400 hover:text-red-600">×</button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
