import {
  useEffect,
  useRef,
  useState,
} from 'react';

import {
  AnimatePresence,
  motion,
} from 'framer-motion';
import {
  Link,
  useSearchParams,
} from 'react-router-dom';

const API_URL =
  "https://bryanhoulton--streaming-memory-tutorservice-serve.modal.run";

const SCENARIOS = {
  tutor: {
    name: "AI Tutor",
    description:
      "You are Alex, a 5th grade student. Your tutor has months of memories about how you learn.",
    placeholder: "Ask about homework, math, or anything...",
    suggestedQuestion: "Can you help me with fractions?",
  },
  dad: {
    name: "Personal Assistant",
    description:
      "Your assistant has access to your memories and notes. Ask for help thinking through decisions.",
    placeholder: "Ask for advice or help planning...",
    suggestedQuestion: "What should I get my dad for his birthday?",
  },
};

export default function Demo() {
  const [searchParams] = useSearchParams();
  const scenario = searchParams.get("scenario") || "tutor";
  const scenarioConfig = SCENARIOS[scenario] || SCENARIOS.tutor;

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentMemories, setCurrentMemories] = useState([]);
  const [thinking, setThinking] = useState("");
  const [memoryUpdates, setMemoryUpdates] = useState(0);
  const [hoveredMemories, setHoveredMemories] = useState(null);
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });
  const [updateFrequency, setUpdateFrequency] = useState(1);
  const [maxMemories, setMaxMemories] = useState(5);
  const [lookbackTokens, setLookbackTokens] = useState(60);
  const [baseContextSize, setBaseContextSize] = useState(0);
  const [currentMemoryTokens, setCurrentMemoryTokens] = useState(0);
  const [ragMemoryTokens, setRagMemoryTokens] = useState(0);
  const [allMemoriesTokens, setAllMemoriesTokens] = useState(0);
  const [uniqueMemoriesCount, setUniqueMemoriesCount] = useState(0);
  const [tokenHistory, setTokenHistory] = useState([]);
  const [showGraph, setShowGraph] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [waitingForFirstToken, setWaitingForFirstToken] = useState(false);
  const chatRef = useRef(null);
  const inputRef = useRef(null);
  const lastMessageRef = useRef(null);

  const hasMessages = messages.length > 0;

  // Scroll to show new message near top when user sends
  const scrollToNewMessage = () => {
    if (lastMessageRef.current && chatRef.current) {
      const containerTop = chatRef.current.getBoundingClientRect().top;
      const messageTop = lastMessageRef.current.getBoundingClientRect().top;
      const offset = messageTop - containerTop - 40;
      chatRef.current.scrollTop += offset;
    }
  };

  // Auto-scroll during streaming
  useEffect(() => {
    if (isStreaming && chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [thinking, currentMemories, messages]);

  const sendMessage = async () => {
    if (!input.trim() || isStreaming) return;

    const userMessage = input.trim();
    setInput("");
    setIsStreaming(true);
    setWaitingForFirstToken(true);
    setThinking("");
    setMemoryUpdates(0);
    setCurrentMemories([]);
    setBaseContextSize(0);
    setCurrentMemoryTokens(0);
    setRagMemoryTokens(0);
    setAllMemoriesTokens(0);
    setUniqueMemoriesCount(0);
    setTokenHistory([]);
    setShowGraph(false);

    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);

    const assistantId = Date.now();
    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content: "",
        id: assistantId,
        streaming: true,
        timeline: [],
        thinkingTimeline: [],
      },
    ]);

    setTimeout(scrollToNewMessage, 50);

    try {
      const history = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const response = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage,
          history,
          update_every_n: updateFrequency,
          max_memories: maxMemories,
          lookback_tokens: lookbackTokens,
          scenario,
        }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullResponse = "";
      let fullThinking = "";
      let timeline = [];
      let thinkingTimeline = [];
      let latestMemories = [];
      let latestTokenHistory = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop();

        for (const event of events) {
          if (!event.trim()) continue;
          const match = event.match(/^data:\s*(.+)$/s);
          if (!match) continue;

          try {
            const data = JSON.parse(match[1]);

            if (data.type === "context_size") {
              if (data.base_context_size)
                setBaseContextSize(data.base_context_size);
              if (data.current_memory_tokens !== undefined)
                setCurrentMemoryTokens(data.current_memory_tokens);
              if (data.rag_memory_tokens !== undefined)
                setRagMemoryTokens(data.rag_memory_tokens);
              if (data.all_memories_tokens !== undefined)
                setAllMemoriesTokens(data.all_memories_tokens);
              if (data.unique_memories)
                setUniqueMemoriesCount(data.unique_memories);
              if (data.token_history) {
                latestTokenHistory = data.token_history;
                setTokenHistory(data.token_history);
              }
            } else if (data.type === "memories") {
              latestMemories = data.memories;
              setCurrentMemories(data.memories);
              setWaitingForFirstToken(false);
            } else if (data.type === "memory_update") {
              latestMemories = data.memories;
              setCurrentMemories(data.memories);
              setMemoryUpdates((prev) => prev + 1);
              if (data.base_context_size)
                setBaseContextSize(data.base_context_size);
              if (data.current_memory_tokens !== undefined)
                setCurrentMemoryTokens(data.current_memory_tokens);
              if (data.rag_memory_tokens !== undefined)
                setRagMemoryTokens(data.rag_memory_tokens);
              if (data.all_memories_tokens !== undefined)
                setAllMemoriesTokens(data.all_memories_tokens);
              if (data.unique_memories)
                setUniqueMemoriesCount(data.unique_memories);
              if (data.token_history) {
                latestTokenHistory = data.token_history;
                setTokenHistory(data.token_history);
              }
            } else if (data.type === "thinking") {
              fullThinking += data.t;
              thinkingTimeline.push({
                token: data.t,
                memories: [...latestMemories],
              });
              setThinking(fullThinking);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, thinkingTimeline: [...thinkingTimeline] }
                    : m
                )
              );
            } else if (data.type === "token") {
              fullResponse += data.t;
              timeline.push({ token: data.t, memories: [...latestMemories] });
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, content: fullResponse, timeline: [...timeline] }
                    : m
                )
              );
            } else if (data.type === "max_tokens") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        hitMaxTokens: true,
                      }
                    : m
                )
              );
            } else if (data.type === "done") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        streaming: false,
                        thinking: fullThinking,
                        timeline: [...timeline],
                        thinkingTimeline: [...thinkingTimeline],
                        tokenHistory: [...latestTokenHistory],
                      }
                    : m
                )
              );
              setThinking("");
              setCurrentMemories([]);
            }
          } catch (e) {
            console.error("Parse error:", e);
          }
        }
      }
    } catch (e) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: `Error: ${e.message}`, streaming: false }
            : m
        )
      );
    }

    setIsStreaming(false);
    inputRef.current?.focus();
  };

  const hoverTimeoutRef = useRef(null);

  const handleTokenHover = (memories, e) => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }
    if (memories && memories.length > 0) {
      setHoveredMemories(memories);
      setHoverPosition({ x: e.clientX, y: e.clientY });
    }
  };

  const handleTokenLeave = () => {
    hoverTimeoutRef.current = setTimeout(() => {
      setHoveredMemories(null);
    }, 150);
  };

  const handleTooltipEnter = () => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }
  };

  const handleTooltipLeave = () => {
    setHoveredMemories(null);
  };

  const renderTokenizedText = (timeline, isThinking = false) => {
    if (!timeline || timeline.length === 0) return null;

    return (
      <span className={isThinking ? "italic" : ""}>
        {timeline.map((item, i) => (
          <span
            key={i}
            className="hover:bg-[#e8e8e8] rounded cursor-pointer transition-colors"
            onMouseEnter={(e) => handleTokenHover(item.memories, e)}
            onMouseLeave={handleTokenLeave}
          >
            {item.token}
          </span>
        ))}
      </span>
    );
  };

  // Empty state - centered input
  if (!hasMessages) {
    return (
      <div className="h-screen overflow-hidden bg-white flex flex-col items-center justify-center px-4 relative">
        {/* Top bar - Back left, Settings right */}
        <div className="absolute top-4 left-4 right-4 flex justify-between items-start">
          <Link
            to="/"
            className="text-xs text-[#999] hover:text-[#666] flex items-center gap-1 transition-colors"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 19l-7-7 7-7"
              />
            </svg>
            Back
          </Link>

          <div className="relative">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="text-xs text-[#999] hover:text-[#666] flex items-center gap-1 transition-colors"
            >
              Settings
              <svg
                className={`w-3 h-3 transition-transform ${
                  showSettings ? "rotate-90" : ""
                }`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5l7 7-7 7"
                />
              </svg>
            </button>

            <AnimatePresence>
              {showSettings && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="absolute top-8 right-0 z-10"
                >
                  <div className="bg-white rounded-lg p-4 space-y-4 text-xs shadow-lg border border-[#eee] w-80">
                    <div className="flex items-center justify-between">
                      <span className="text-[#666]">Update frequency</span>
                      <div className="flex items-center gap-3">
                        <input
                          type="range"
                          min="1"
                          max="20"
                          value={updateFrequency}
                          onChange={(e) =>
                            setUpdateFrequency(Number(e.target.value))
                          }
                          className="w-16 h-1 bg-[#e5e5e5] rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[#666] [&::-webkit-slider-thumb]:rounded-full"
                        />
                        <span className="w-20 text-[#999] text-right">
                          {updateFrequency} token
                          {updateFrequency > 1 ? "s" : ""}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-[#666]">Max memories</span>
                      <div className="flex items-center gap-3">
                        <input
                          type="range"
                          min="1"
                          max="15"
                          value={maxMemories}
                          onChange={(e) =>
                            setMaxMemories(Number(e.target.value))
                          }
                          className="w-16 h-1 bg-[#e5e5e5] rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[#666] [&::-webkit-slider-thumb]:rounded-full"
                        />
                        <span className="w-20 text-[#999] text-right">
                          {maxMemories}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-[#666]">Lookback window</span>
                      <div className="flex items-center gap-3">
                        <input
                          type="range"
                          min="10"
                          max="150"
                          step="10"
                          value={lookbackTokens}
                          onChange={(e) =>
                            setLookbackTokens(Number(e.target.value))
                          }
                          className="w-16 h-1 bg-[#e5e5e5] rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[#666] [&::-webkit-slider-thumb]:rounded-full"
                        />
                        <span className="w-20 text-[#999] text-right">
                          {lookbackTokens} tokens
                        </span>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        <h1 className="text-2xl font-bold text-[#1a1a1a] mb-2">
          {scenarioConfig.name}
        </h1>
        <p className="text-[#999] text-sm mb-6 max-w-md text-center">
          {scenarioConfig.description}
        </p>

        {/* Scenario switcher */}
        <div className="flex gap-2 mb-6">
          {Object.entries(SCENARIOS).map(([key, config]) => (
            <Link
              key={key}
              to={`/demo?scenario=${key}`}
              className={`px-4 py-2 rounded-full text-sm transition-colors ${
                scenario === key
                  ? "bg-[#1a1a1a] text-white"
                  : "bg-[#f5f5f5] text-[#666] hover:bg-[#eee]"
              }`}
            >
              {config.name}
            </Link>
          ))}
        </div>

        <div className="w-full max-w-2xl">
          <div className="relative">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) =>
                e.key === "Enter" && !e.shiftKey && sendMessage()
              }
              placeholder={scenarioConfig.placeholder}
              autoFocus
              className="w-full px-4 py-3 pr-12 rounded-full bg-[#f5f5f5] text-[#1a1a1a] placeholder-[#999] focus:outline-none focus:ring-2 focus:ring-[#ddd]"
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim()}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-1.5 rounded-full text-[#999] hover:text-[#666] disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M14 5l7 7m0 0l-7 7m7-7H3"
                />
              </svg>
            </button>
          </div>

          {/* Suggested question - centered */}
          <div className="text-center mt-3">
            <button
              onClick={() => setInput(scenarioConfig.suggestedQuestion)}
              className="text-sm text-[#999] hover:text-[#666] transition-colors"
            >
              Try: "{scenarioConfig.suggestedQuestion}"
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Chat view - input at bottom
  return (
    <div className="h-screen overflow-hidden bg-white flex flex-col">
      {/* Top bar - Back left, Settings right */}
      <div className="flex-shrink-0 px-4 pt-3 flex justify-between items-start">
        <Link
          to="/"
          className="text-xs text-[#999] hover:text-[#666] flex items-center gap-1 transition-colors"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 19l-7-7 7-7"
            />
          </svg>
          Back
        </Link>

        <div className="relative">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="text-xs text-[#999] hover:text-[#666] flex items-center gap-1 transition-colors"
          >
            <svg
              className={`w-3 h-3 transition-transform ${
                showSettings ? "rotate-90" : ""
              }`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 5l7 7-7 7"
              />
            </svg>
            Settings
          </button>

          <AnimatePresence>
            {showSettings && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute top-8 right-0 z-10"
              >
                <div className="bg-white rounded-lg p-4 space-y-4 text-xs shadow-lg border border-[#eee] w-80">
                  <div className="flex items-center justify-between">
                    <span className="text-[#666]">Update frequency</span>
                    <div className="flex items-center gap-3">
                      <input
                        type="range"
                        min="1"
                        max="20"
                        value={updateFrequency}
                        onChange={(e) =>
                          setUpdateFrequency(Number(e.target.value))
                        }
                        disabled={isStreaming}
                        className="w-16 h-1 bg-[#e5e5e5] rounded-full appearance-none cursor-pointer disabled:opacity-50 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[#666] [&::-webkit-slider-thumb]:rounded-full"
                      />
                      <span className="w-20 text-[#999] text-right">
                        {updateFrequency} token{updateFrequency > 1 ? "s" : ""}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-[#666]">Max memories</span>
                    <div className="flex items-center gap-3">
                      <input
                        type="range"
                        min="1"
                        max="15"
                        value={maxMemories}
                        onChange={(e) => setMaxMemories(Number(e.target.value))}
                        disabled={isStreaming}
                        className="w-16 h-1 bg-[#e5e5e5] rounded-full appearance-none cursor-pointer disabled:opacity-50 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[#666] [&::-webkit-slider-thumb]:rounded-full"
                      />
                      <span className="w-20 text-[#999] text-right">
                        {maxMemories}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-[#666]">Lookback window</span>
                    <div className="flex items-center gap-3">
                      <input
                        type="range"
                        min="10"
                        max="150"
                        step="10"
                        value={lookbackTokens}
                        onChange={(e) =>
                          setLookbackTokens(Number(e.target.value))
                        }
                        disabled={isStreaming}
                        className="w-16 h-1 bg-[#e5e5e5] rounded-full appearance-none cursor-pointer disabled:opacity-50 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[#666] [&::-webkit-slider-thumb]:rounded-full"
                      />
                      <span className="w-20 text-[#999] text-right">
                        {lookbackTokens} tokens
                      </span>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Chat History */}
      <div ref={chatRef} className="flex-1 overflow-y-auto px-4">
        <div className="max-w-2xl mx-auto py-6 space-y-6">
          {messages.map((msg, i) => {
            const isLastUserMessage =
              msg.role === "user" && i === messages.length - 2;
            return (
              <div key={i} ref={isLastUserMessage ? lastMessageRef : null}>
                {/* User message */}
                {msg.role === "user" && (
                  <div className="flex justify-end mb-4">
                    <div className="bg-[#f5f5f5] rounded-2xl px-4 py-3 max-w-[80%]">
                      <p className="text-[#1a1a1a]">{msg.content}</p>
                    </div>
                  </div>
                )}

                {/* Assistant message */}
                {msg.role === "assistant" && (
                  <div className="mb-4">
                    {/* Cold start / waiting indicator */}
                    {msg.streaming && waitingForFirstToken && (
                      <div className="text-[#aaa] text-sm mb-3 pl-1 flex items-center gap-2">
                        <svg
                          className="w-4 h-4 animate-spin"
                          fill="none"
                          viewBox="0 0 24 24"
                        >
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="3"
                          ></circle>
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                          ></path>
                        </svg>
                        <span className="italic">Waking up...</span>
                      </div>
                    )}

                    {/* Thinking */}
                    {(msg.thinkingTimeline?.length > 0 ||
                      (msg.streaming && thinking && !waitingForFirstToken)) && (
                      <div className="text-[#aaa] text-sm mb-3 pl-1">
                        {msg.thinkingTimeline?.length > 0 ? (
                          renderTokenizedText(msg.thinkingTimeline, true)
                        ) : (
                          <span className="italic">{thinking}</span>
                        )}
                      </div>
                    )}

                    {/* Response */}
                    <div className="pl-1">
                      <p className="text-[#1a1a1a] whitespace-pre-wrap">
                        {msg.timeline?.length > 0
                          ? renderTokenizedText(msg.timeline)
                          : msg.content}
                        {msg.streaming && <span className="cursor" />}
                      </p>
                      {msg.hitMaxTokens && (
                        <p className="text-orange-500 text-sm mt-2 italic">
                          ⚠️ Response truncated (hit token limit)
                        </p>
                      )}
                      {/* Graph button for completed messages */}
                      {!msg.streaming && msg.tokenHistory?.length > 0 && (
                        <button
                          onClick={() => {
                            setTokenHistory(msg.tokenHistory);
                            setShowGraph(true);
                          }}
                          className="mt-2 text-xs text-[#999] hover:text-[#666] flex items-center gap-1 transition-colors"
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                          </svg>
                          View token usage
                        </button>
                      )}
                    </div>

                    {/* Active Memories */}
                    {msg.streaming && currentMemories.length > 0 && (
                      <div className="mt-4 pl-1">
                        <div className="text-xs mb-2 space-y-2">
                          <div className="flex items-center gap-2 text-[#999]">
                            <span>Active memories</span>
                            {memoryUpdates > 0 && (
                              <span className="text-[#bbb]">
                                ({memoryUpdates} swaps)
                              </span>
                            )}
                          </div>
                          {allMemoriesTokens > 0 && (
                            <div className="bg-[#f8f8f8] rounded-lg p-2 space-y-1.5">
                              <div className="text-[#999]">
                                {uniqueMemoriesCount} memories retrieved this run
                              </div>
                              <div className="flex items-center justify-between text-[#666]">
                                <span>All memories:</span>
                                <span className="font-mono text-red-400">
                                  {(baseContextSize + allMemoriesTokens).toLocaleString()} tokens
                                </span>
                              </div>
                              <div className="flex items-center justify-between text-[#666]">
                                <span>RAG (retrieved once):</span>
                                <span className="font-mono text-orange-400">
                                  {(baseContextSize + ragMemoryTokens).toLocaleString()} tokens
                                </span>
                              </div>
                              <div className="flex items-center justify-between text-[#666]">
                                <span>Streaming memory:</span>
                                <span className="font-mono text-green-600">
                                  {(baseContextSize + currentMemoryTokens).toLocaleString()} tokens
                                </span>
                              </div>
                            </div>
                          )}
                        </div>
                        <div className="relative">
                          <div className="space-y-2 max-h-40 overflow-y-auto pr-2 pb-12">
                            {currentMemories.map((mem, j) => (
                              <motion.div
                                key={mem}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: j * 0.02 }}
                                className="text-xs text-[#888] bg-[#fafafa] rounded-lg px-3 py-2"
                              >
                                {mem}
                              </motion.div>
                            ))}
                          </div>
                          <div className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-white to-transparent pointer-events-none" />
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Input Area */}
      <div className="flex-shrink-0 bg-white px-4 py-4">
        <div className="max-w-2xl mx-auto">
          <div className="relative">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) =>
                e.key === "Enter" && !e.shiftKey && sendMessage()
              }
              placeholder="Message..."
              disabled={isStreaming}
              className="w-full px-4 py-3 pr-12 rounded-full bg-[#f5f5f5] text-[#1a1a1a] placeholder-[#999] focus:outline-none focus:ring-2 focus:ring-[#ddd] disabled:bg-[#e8e8e8] disabled:cursor-not-allowed"
            />
            <button
              onClick={sendMessage}
              disabled={isStreaming || !input.trim()}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-1.5 rounded-full text-[#999] hover:text-[#666] disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              {isStreaming ? (
                <svg
                  className="w-5 h-5 animate-spin"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="3"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
              ) : (
                <svg
                  className="w-5 h-5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M14 5l7 7m0 0l-7 7m7-7H3"
                  />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Hover tooltip */}
      <AnimatePresence>
        {hoveredMemories && (
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 5 }}
            className="fixed z-50 max-w-sm bg-white rounded-lg shadow-lg p-3"
            style={{
              left: Math.min(hoverPosition.x + 10, window.innerWidth - 350),
              top: hoverPosition.y + 20,
            }}
            onMouseEnter={handleTooltipEnter}
            onMouseLeave={handleTooltipLeave}
          >
            <div className="text-xs text-[#999] mb-2">
              Memories at this token:
            </div>
            <div className="space-y-1.5 max-h-48 overflow-y-auto">
              {hoveredMemories.map((mem, i) => (
                <div
                  key={i}
                  className="text-xs text-[#666] bg-[#f5f5f5] rounded px-2 py-1.5"
                >
                  {mem}
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Token usage graph modal */}
      <AnimatePresence>
        {showGraph && tokenHistory.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/30"
            onClick={() => setShowGraph(false)}
          >
            <motion.div
              initial={{ scale: 0.95 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.95 }}
              className="bg-white rounded-xl shadow-xl p-6 max-w-2xl w-full mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-[#1a1a1a]">
                  Token Usage Over Generation
                </h3>
                <button
                  onClick={() => setShowGraph(false)}
                  className="text-[#999] hover:text-[#666] transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              {/* SVG Graph */}
              <div className="relative h-64 mt-4">
                <svg viewBox="0 0 500 200" className="w-full h-full" preserveAspectRatio="none">
                  {/* Y-axis labels */}
                  {(() => {
                    const maxVal = Math.max(...tokenHistory.map(h => h.all));
                    const steps = [0, 0.25, 0.5, 0.75, 1];
                    return steps.map((step, i) => (
                      <text
                        key={i}
                        x="0"
                        y={200 - step * 180 - 10}
                        className="text-[8px] fill-[#999]"
                      >
                        {Math.round(maxVal * step).toLocaleString()}
                      </text>
                    ));
                  })()}
                  
                  {/* Grid lines */}
                  <line x1="45" y1="10" x2="45" y2="190" stroke="#eee" strokeWidth="1" />
                  <line x1="45" y1="190" x2="495" y2="190" stroke="#eee" strokeWidth="1" />
                  {[0.25, 0.5, 0.75].map((step, i) => (
                    <line key={i} x1="45" y1={190 - step * 180} x2="495" y2={190 - step * 180} stroke="#f5f5f5" strokeWidth="1" />
                  ))}
                  
                  {/* Lines */}
                  {(() => {
                    const maxVal = Math.max(...tokenHistory.map(h => h.all));
                    const xScale = 450 / (tokenHistory.length - 1 || 1);
                    const yScale = 180 / maxVal;
                    
                    const allPath = tokenHistory.map((h, i) => 
                      `${i === 0 ? 'M' : 'L'} ${45 + i * xScale} ${190 - h.all * yScale}`
                    ).join(' ');
                    
                    const ragPath = tokenHistory.map((h, i) => 
                      `${i === 0 ? 'M' : 'L'} ${45 + i * xScale} ${190 - h.rag * yScale}`
                    ).join(' ');
                    
                    const streamingPath = tokenHistory.map((h, i) => 
                      `${i === 0 ? 'M' : 'L'} ${45 + i * xScale} ${190 - h.streaming * yScale}`
                    ).join(' ');
                    
                    return (
                      <>
                        <path d={allPath} fill="none" stroke="#f87171" strokeWidth="2" />
                        <path d={ragPath} fill="none" stroke="#fb923c" strokeWidth="2" />
                        <path d={streamingPath} fill="none" stroke="#22c55e" strokeWidth="2" />
                      </>
                    );
                  })()}
                </svg>
              </div>
              
              {/* Legend */}
              <div className="flex gap-6 justify-center mt-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-red-400" />
                  <span className="text-[#666]">All memories</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-orange-400" />
                  <span className="text-[#666]">RAG (retrieved once)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-green-500" />
                  <span className="text-[#666]">Streaming memory</span>
                </div>
              </div>
              
              {/* Summary */}
              {tokenHistory.length > 0 && (
                <div className="mt-4 pt-4 border-t border-[#eee] text-center text-sm text-[#666]">
                  <span className="text-green-600 font-medium">
                    {Math.round((1 - tokenHistory[tokenHistory.length - 1].streaming / tokenHistory[tokenHistory.length - 1].all) * 100)}% fewer tokens
                  </span>
                  {' '}than including all memories
                </div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
