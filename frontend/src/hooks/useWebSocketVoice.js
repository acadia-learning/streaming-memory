import { useEffect, useRef, useState, useCallback } from 'react';

/**
 * useWebSocketVoice - WebSocket voice agent hook
 */

export function useWebSocketVoice(serverUrl) {
  const wsRef = useRef(null);
  const [stage, setStage] = useState('idle');

  // Audio capture
  const audioContextRef = useRef(null);
  const streamRef = useRef(null);
  const processorRef = useRef(null);
  const sourceRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const isRecordingRef = useRef(false);

  // State
  const [fullTranscript, setFullTranscript] = useState('');
  const [displayedTranscript, setDisplayedTranscript] = useState('');
  const [currentThinking, setCurrentThinking] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [thinkingCommitted, setThinkingCommitted] = useState(false);  // True after EndOfTurn - don't clear
  const thinkingCommittedRef = useRef(false);  // Ref for callback access
  const [currentResponse, setCurrentResponse] = useState('');
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [responseLatency, setResponseLatency] = useState(null);
  const [userSpeaking, setUserSpeaking] = useState(false);
  const [error, setError] = useState(null);
  const [eagerEndOfTurn, setEagerEndOfTurn] = useState(false);  // Debug: show when EagerEndOfTurn fires
  const [modelContext, setModelContext] = useState(null);  // Debug: show model's context
  const [thinkingStalled, setThinkingStalled] = useState(false);  // Debug: show when model stalls while user speaking
  const [stallInfo, setStallInfo] = useState(null);  // Info about the stall

  // TTS
  const audioQueueRef = useRef([]);
  const isPlayingRef = useRef(false);
  const playbackCtxRef = useRef(null);

  // Debug counter
  const chunksSentRef = useRef(0);

  // Simple linear resampler
  const resample = (inputBuffer, inputSampleRate, outputSampleRate) => {
    if (inputSampleRate === outputSampleRate) {
      return inputBuffer;
    }
    const ratio = inputSampleRate / outputSampleRate;
    const outputLength = Math.floor(inputBuffer.length / ratio);
    const output = new Float32Array(outputLength);
    for (let i = 0; i < outputLength; i++) {
      const srcIndex = i * ratio;
      const srcIndexFloor = Math.floor(srcIndex);
      const srcIndexCeil = Math.min(srcIndexFloor + 1, inputBuffer.length - 1);
      const t = srcIndex - srcIndexFloor;
      output[i] = inputBuffer[srcIndexFloor] * (1 - t) + inputBuffer[srcIndexCeil] * t;
    }
    return output;
  };

  // Convert Float32 to PCM16 base64
  const float32ToPcm16Base64 = (float32Array) => {
    const pcm16 = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    const uint8 = new Uint8Array(pcm16.buffer);
    let binary = '';
    for (let i = 0; i < uint8.length; i++) {
      binary += String.fromCharCode(uint8[i]);
    }
    return btoa(binary);
  };

  // Play TTS audio
  const playNextChunk = useCallback(async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) return;

    isPlayingRef.current = true;
    const base64 = audioQueueRef.current.shift();

    try {
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }

      if (!playbackCtxRef.current) {
        playbackCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      const ctx = playbackCtxRef.current;

      const pcm16 = new Int16Array(bytes.buffer);
      const float32 = new Float32Array(pcm16.length);
      for (let i = 0; i < pcm16.length; i++) {
        float32[i] = pcm16[i] / 32768;
      }

      const buffer = ctx.createBuffer(1, float32.length, 24000);
      buffer.getChannelData(0).set(float32);

      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.onended = () => {
        isPlayingRef.current = false;
        playNextChunk();
      };
      source.start();
    } catch (err) {
      console.error('[Voice] TTS error:', err);
      isPlayingRef.current = false;
      playNextChunk();
    }
  }, []);

  // Handle server messages
  const handleMessage = useCallback((event) => {
    try {
      const msg = JSON.parse(event.data);
      const { type } = msg;

      switch (type) {
        case 'connected':
          setStage('connected');
          break;
        case 'status':
          break;
        case 'transcript':
          if (msg.is_final) {
            setFullTranscript(msg.full_transcript);
            setDisplayedTranscript(msg.full_transcript);
            setUserSpeaking(false);
          } else {
            setDisplayedTranscript(msg.full_transcript);
            setUserSpeaking(true);
          }
          break;
        case 'thinking_start':
          setIsThinking(true);
          // Only clear if thinking isn't committed
          if (!thinkingCommittedRef.current) {
            setCurrentThinking('');
          }
          break;
        case 'thinking_end':
          setIsThinking(false);
          setThinkingStalled(false);
          setStallInfo(null);
          if (msg.latency_ms) setResponseLatency(msg.latency_ms);
          break;
        case 'token':
          if (msg.is_thinking) {
            setCurrentThinking(prev => prev + msg.t);
          } else {
            setCurrentResponse(prev => prev + msg.t);
          }
          break;
        case 'restart_thinking':
          // Ignore restart_thinking if thinking is committed (user finished speaking)
          if (thinkingCommittedRef.current) {
            break;
          }
          setCurrentThinking('');
          setCurrentResponse('');
          setIsThinking(true);
          setEagerEndOfTurn(false);
          setThinkingStalled(false);
          setStallInfo(null);
          break;
        case 'EndOfTurn':
          setEagerEndOfTurn(true);
          setThinkingCommitted(true);
          thinkingCommittedRef.current = true;  // Commit thinking - don't clear anymore
          setTimeout(() => setEagerEndOfTurn(false), 2000);
          break;
        case 'context':
          setModelContext(msg.prompt);
          break;
        case 'thinking_stalled':
          console.warn('[Voice] THINKING STALLED:', msg);
          setThinkingStalled(true);
          setStallInfo(msg);
          break;
        case 'thinking_unstalled':
          console.log('[Voice] Thinking unstalled:', msg);
          setThinkingStalled(false);
          setStallInfo(null);
          break;
        case 'response_complete':
          setIsSpeaking(false);
          // Reset for next turn
          setThinkingCommitted(false);
          thinkingCommittedRef.current = false;
          setCurrentThinking('');
          setCurrentResponse('');
          break;
        case 'tts_start':
          setIsSpeaking(true);
          // Don't clear queue if already playing - might lose audio
          if (!isPlayingRef.current) {
            audioQueueRef.current = [];
          }
          break;
        case 'tts_audio':
          audioQueueRef.current.push(msg.audio);
          playNextChunk();
          break;
        case 'tts_end':
          break;
        default:
          break;
      }
    } catch (err) {
      console.error('[Voice] Parse error:', err);
    }
  }, [playNextChunk]);

  // Connect
  const connect = useCallback(async () => {
    if (stage !== 'idle') return;

    setStage('connecting');
    setError(null);

    try {
      const wsUrl = serverUrl.replace('https://', 'wss://').replace('http://', 'ws://') + '/ws';
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        wsRef.current = ws;
      };

      ws.onmessage = handleMessage;

      ws.onerror = (err) => {
        console.error('[Voice] WebSocket ERROR:', err);
        setError('Connection error');
        setStage('error');
      };

      ws.onclose = () => {
        wsRef.current = null;
        setStage('idle');
        setIsRecording(false);
        isRecordingRef.current = false;
      };
    } catch (err) {
      console.error('[Voice] Connect error:', err);
      setError(err.message);
      setStage('error');
    }
  }, [stage, serverUrl, handleMessage]);

  // Disconnect
  const disconnect = useCallback(() => {
    if (wsRef.current) wsRef.current.close();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
    }
  }, []);

  // Toggle recording
  const toggleRecording = useCallback(async () => {
    if (!isRecording) {
      // START RECORDING
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
        streamRef.current = stream;

        // Create audio context
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        const audioContext = new AudioContextClass();
        audioContextRef.current = audioContext;
        
        // Resume if suspended (required for user gesture)
        if (audioContext.state === 'suspended') {
          await audioContext.resume();
        }
        
        const sampleRate = audioContext.sampleRate;

        // Create source from mic
        const source = audioContext.createMediaStreamSource(stream);
        sourceRef.current = source;

        // Create processor
        const processor = audioContext.createScriptProcessor(4096, 1, 1);
        processorRef.current = processor;
        
        // Reset counter
        chunksSentRef.current = 0;

        // Audio processing callback
        processor.onaudioprocess = (e) => {
          // Check if we should be recording
          if (!isRecordingRef.current) {
            return;
          }
          
          // Check WebSocket
          if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.warn('[Voice] WebSocket not ready, skipping audio chunk');
            return;
          }

          // Get audio data
          const inputData = e.inputBuffer.getChannelData(0);
          
          // Resample to 16kHz
          const resampled = resample(inputData, sampleRate, 16000);
          
          // Convert to base64 PCM16
          const base64 = float32ToPcm16Base64(resampled);
          
          // Send to server
          try {
            wsRef.current.send(JSON.stringify({ type: 'audio', audio: base64 }));
            chunksSentRef.current++;
            
          } catch (err) {
            console.error('[Voice] Error sending audio:', err);
          }
        };

        // Connect the audio graph
        source.connect(processor);
        processor.connect(audioContext.destination);

        // Set recording state AFTER everything is set up
        isRecordingRef.current = true;
        setIsRecording(true);
        
      } catch (err) {
        console.error('[Voice] Recording error:', err);
        setError(`Microphone error: ${err.message}`);
      }
    } else {
      // STOP RECORDING
      isRecordingRef.current = false;
      
      if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
      }
      if (sourceRef.current) {
        sourceRef.current.disconnect();
        sourceRef.current = null;
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      }

      setIsRecording(false);
    }
  }, [isRecording]);

  // Cleanup
  useEffect(() => {
    return () => {
      disconnect();
      if (audioContextRef.current) audioContextRef.current.close();
      if (playbackCtxRef.current) playbackCtxRef.current.close();
    };
  }, [disconnect]);

  return {
    stage,
    connect,
    disconnect,
    error,
    setError,
    isRecording,
    toggleRecording,
    fullTranscript,
    displayedTranscript,
    userSpeaking,
    currentThinking,
    isThinking,
    thinkingCommitted,
    currentResponse,
    isSpeaking,
    responseLatency,
    eagerEndOfTurn,
    modelContext,
    thinkingStalled,
    stallInfo,
  };
}
