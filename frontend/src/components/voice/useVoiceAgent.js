import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Room,
  RoomEvent,
  createLocalAudioTrack,
} from 'livekit-client';

const JOIN_URL = import.meta.env.VITE_JOIN_URL || 'https://bryanhoulton--livekit-voice-agent-join-room.modal.run';

export default function useVoiceAgent() {
  // Connection state
  const [stage, setStage] = useState('idle');
  const [loadingStep, setLoadingStep] = useState(0);
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // Content state
  const [liveTranscript, setLiveTranscript] = useState('');
  const [displayedTranscript, setDisplayedTranscript] = useState('');
  const [currentThinking, setCurrentThinking] = useState('');
  const [currentResponse, setCurrentResponse] = useState('');
  const [isThinking, setIsThinking] = useState(true);
  const [userSpeaking, setUserSpeaking] = useState(false);
  const [responseLatency, setResponseLatency] = useState(null);

  // Refs
  const roomRef = useRef(null);
  const audioTrackRef = useRef(null);
  const transcriptTypingRef = useRef(null);
  const audioContextRef = useRef(null);
  const audioQueueRef = useRef([]);
  const isPlayingRef = useRef(false);

  // Smooth transcription typing
  useEffect(() => {
    if (transcriptTypingRef.current) clearTimeout(transcriptTypingRef.current);
    if (liveTranscript === displayedTranscript) return;

    const backlog = liveTranscript.length - displayedTranscript.length;
    const delay = Math.max(5, Math.min(20, 20 - (backlog / 10)));

    const typeNext = () => {
      if (displayedTranscript.length < liveTranscript.length) {
        setDisplayedTranscript(liveTranscript.slice(0, displayedTranscript.length + 1));
        transcriptTypingRef.current = setTimeout(typeNext, delay);
      }
    };
    typeNext();

    return () => {
      if (transcriptTypingRef.current) clearTimeout(transcriptTypingRef.current);
    };
  }, [liveTranscript, displayedTranscript]);

  // Play queued audio chunks
  const playNextAudioChunk = useCallback(async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) {
      if (audioQueueRef.current.length === 0) {
        setIsSpeaking(false);
      }
      return;
    }

    isPlayingRef.current = true;
    setIsSpeaking(true);
    
    const audioData = audioQueueRef.current.shift();
    
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
      }
      
      const ctx = audioContextRef.current;
      
      const binaryString = atob(audioData);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      const int16 = new Int16Array(bytes.buffer);
      const float32 = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; i++) {
        float32[i] = int16[i] / 32768.0;
      }
      
      const audioBuffer = ctx.createBuffer(1, float32.length, 24000);
      audioBuffer.getChannelData(0).set(float32);
      
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      source.onended = () => {
        isPlayingRef.current = false;
        playNextAudioChunk();
      };
      source.start();
    } catch (e) {
      console.error('Audio playback error:', e);
      isPlayingRef.current = false;
      playNextAudioChunk();
    }
  }, []);

  // Handle data from agent
  const handleAgentData = useCallback((data) => {
    try {
      const event = JSON.parse(new TextDecoder().decode(data));
      
      switch (event.type) {
        case 'status':
          if (event.stage === 'container') setLoadingStep(1);
          else if (event.stage === 'connecting') setLoadingStep(2);
          else if (event.stage === 'model') setLoadingStep(2);
          else if (event.stage === 'ready') {
            setLoadingStep(3);
            setTimeout(() => setStage('active'), 500);
          }
          break;

        case 'transcript':
          setLiveTranscript(event.full_transcript || event.text);
          setUserSpeaking(event.user_speaking);
          break;

        case 'generation_start':
          setCurrentThinking('');
          setCurrentResponse('');
          setResponseLatency(null);
          setIsThinking(true);
          break;

        case 'restart_thinking':
          setCurrentThinking('');
          setCurrentResponse('');
          setResponseLatency(null);
          setIsThinking(true);
          audioQueueRef.current = [];
          setIsSpeaking(false);
          break;

        case 'thinking_end':
          setIsThinking(false);
          if (event.latency_ms) setResponseLatency(event.latency_ms);
          break;

        case 'token':
          if (event.is_thinking) {
            setCurrentThinking((prev) => prev + event.t);
          } else {
            setCurrentResponse((prev) => prev + event.t);
          }
          break;

        case 'tts_start':
          audioQueueRef.current = [];
          break;
          
        case 'tts_audio':
          if (event.audio) {
            audioQueueRef.current.push(event.audio);
            playNextAudioChunk();
          }
          break;
          
        case 'tts_end':
          break;

        default:
          break;
      }
    } catch (e) {
      console.error('Failed to parse agent data:', e);
    }
  }, [playNextAudioChunk]);

  // Connect to LiveKit
  const connect = useCallback(async () => {
    if (roomRef.current) return;

    try {
      setStage('loading');
      setLoadingStep(1);
      setError(null);

      const response = await fetch(JOIN_URL);
      if (!response.ok) throw new Error('Failed to start session');
      const { token, room: roomName, livekit_url } = await response.json();

      const room = new Room();
      roomRef.current = room;

      room.on(RoomEvent.Connected, () => {
        setIsConnected(true);
      });

      room.on(RoomEvent.Disconnected, () => {
        setIsConnected(false);
        setIsRecording(false);
        setStage('idle');
        roomRef.current = null;
      });

      room.on(RoomEvent.DataReceived, (data, participant) => {
        if (participant?.identity === 'voice-agent') {
          handleAgentData(data);
        }
      });

      await room.connect(livekit_url, token);

      const audioTrack = await createLocalAudioTrack({
        echoCancellation: true,
        noiseSuppression: true,
        sampleRate: 16000,
      });
      audioTrackRef.current = audioTrack;
      await room.localParticipant.publishTrack(audioTrack);
      setIsRecording(true);

    } catch (e) {
      console.error('Connection error:', e);
      setError(`Connection failed: ${e.message}`);
      setStage('idle');
    }
  }, [handleAgentData]);

  // Toggle recording
  const toggleRecording = useCallback(async () => {
    if (isRecording) {
      if (audioTrackRef.current) {
        audioTrackRef.current.stop();
        if (roomRef.current) {
          await roomRef.current.localParticipant.unpublishTrack(audioTrackRef.current);
        }
        audioTrackRef.current = null;
      }
      setIsRecording(false);
    } else {
      if (!roomRef.current || !isConnected) return;
      try {
        const audioTrack = await createLocalAudioTrack({
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
        });
        audioTrackRef.current = audioTrack;
        await roomRef.current.localParticipant.publishTrack(audioTrack);
        setIsRecording(true);
      } catch (e) {
        setError(`Mic error: ${e.message}`);
      }
    }
  }, [isRecording, isConnected]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (audioTrackRef.current) audioTrackRef.current.stop();
      if (roomRef.current) roomRef.current.disconnect();
      if (audioContextRef.current) audioContextRef.current.close();
    };
  }, []);

  return {
    // State
    stage,
    loadingStep,
    isRecording,
    error,
    isSpeaking,
    liveTranscript,
    displayedTranscript,
    currentThinking,
    currentResponse,
    isThinking,
    userSpeaking,
    responseLatency,
    // Actions
    connect,
    toggleRecording,
    setError,
  };
}



