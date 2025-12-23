import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Landing from './Landing'

const API_URL = 'https://bryanhoulton--streaming-memory-tutorservice-serve.modal.run'

export default function App() {
  const [showDemo, setShowDemo] = useState(false)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [currentMemories, setCurrentMemories] = useState([])
  const [thinking, setThinking] = useState('')
  const [memoryUpdates, setMemoryUpdates] = useState(0)
  const [hoveredMemories, setHoveredMemories] = useState(null)
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 })
  const [updateFrequency, setUpdateFrequency] = useState(1)
  const [maxMemories, setMaxMemories] = useState(5)
  const [showSettings, setShowSettings] = useState(false)
  const chatRef = useRef(null)
  const inputRef = useRef(null)
  const lastMessageRef = useRef(null)

  const hasMessages = messages.length > 0
  
  // Show landing page first
  if (!showDemo) {
    return <Landing onStartDemo={() => setShowDemo(true)} />
  }

  // Scroll to show new message near top when user sends
  const scrollToNewMessage = () => {
    if (lastMessageRef.current && chatRef.current) {
      const containerTop = chatRef.current.getBoundingClientRect().top
      const messageTop = lastMessageRef.current.getBoundingClientRect().top
      const offset = messageTop - containerTop - 40
      chatRef.current.scrollTop += offset
    }
  }

  // Auto-scroll during streaming
  useEffect(() => {
    if (isStreaming && chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight
    }
  }, [thinking, currentMemories, messages])

  const sendMessage = async () => {
    if (!input.trim() || isStreaming) return

    const userMessage = input.trim()
    setInput('')
    setIsStreaming(true)
    setThinking('')
    setMemoryUpdates(0)
    setCurrentMemories([])

    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    
    const assistantId = Date.now()
    setMessages(prev => [...prev, { 
      role: 'assistant', 
      content: '', 
      id: assistantId, 
      streaming: true,
      timeline: [],
      thinkingTimeline: []
    }])

    setTimeout(scrollToNewMessage, 50)

    try {
      const history = messages.map(m => ({ role: m.role, content: m.content }))
      
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage, history, update_every_n: updateFrequency, max_memories: maxMemories })
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let fullResponse = ''
      let fullThinking = ''
      let timeline = []
      let thinkingTimeline = []
      let latestMemories = [] // Track memories locally, not via React state

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const events = buffer.split('\n\n')
        buffer = events.pop()

        for (const event of events) {
          if (!event.trim()) continue
          const match = event.match(/^data:\s*(.+)$/s)
          if (!match) continue

          try {
            const data = JSON.parse(match[1])

            if (data.type === 'memories') {
              latestMemories = data.memories
              setCurrentMemories(data.memories)
            } else if (data.type === 'memory_update') {
              latestMemories = data.memories
              setCurrentMemories(data.memories)
              setMemoryUpdates(prev => prev + 1)
            } else if (data.type === 'thinking') {
              fullThinking += data.t
              thinkingTimeline.push({ token: data.t, memories: [...latestMemories] })
              setThinking(fullThinking)
              setMessages(prev => prev.map(m => 
                m.id === assistantId ? { ...m, thinkingTimeline: [...thinkingTimeline] } : m
              ))
            } else if (data.type === 'token') {
              fullResponse += data.t
              timeline.push({ token: data.t, memories: [...latestMemories] })
              setMessages(prev => prev.map(m => 
                m.id === assistantId ? { ...m, content: fullResponse, timeline: [...timeline] } : m
              ))
            } else if (data.type === 'done') {
              setMessages(prev => prev.map(m => 
                m.id === assistantId ? { 
                  ...m, 
                  streaming: false, 
                  thinking: fullThinking,
                  timeline: [...timeline],
                  thinkingTimeline: [...thinkingTimeline]
                } : m
              ))
              setThinking('')
              setCurrentMemories([])
            }
          } catch (e) {
            console.error('Parse error:', e)
          }
        }
      }
    } catch (e) {
      setMessages(prev => prev.map(m => 
        m.id === assistantId ? { ...m, content: `Error: ${e.message}`, streaming: false } : m
      ))
    }

    setIsStreaming(false)
    inputRef.current?.focus()
  }

  const hoverTimeoutRef = useRef(null)

  const handleTokenHover = (memories, e) => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current)
      hoverTimeoutRef.current = null
    }
    if (memories && memories.length > 0) {
      setHoveredMemories(memories)
      setHoverPosition({ x: e.clientX, y: e.clientY })
    }
  }

  const handleTokenLeave = () => {
    // Delay hiding so user can move to tooltip
    hoverTimeoutRef.current = setTimeout(() => {
      setHoveredMemories(null)
    }, 150)
  }

  const handleTooltipEnter = () => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current)
      hoverTimeoutRef.current = null
    }
  }

  const handleTooltipLeave = () => {
    setHoveredMemories(null)
  }

  const renderTokenizedText = (timeline, isThinking = false) => {
    if (!timeline || timeline.length === 0) return null
    
    return (
      <span className={isThinking ? 'italic' : ''}>
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
    )
  }

  // Empty state - centered input
  if (!hasMessages) {
    return (
      <div className="h-screen overflow-hidden bg-white flex flex-col items-center justify-center px-4">
        <button
          onClick={() => setShowDemo(false)}
          className="absolute top-4 left-4 text-xs text-[#999] hover:text-[#666] flex items-center gap-1 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back
        </button>
        
        <h1 className="text-2xl font-bold text-[#1a1a1a] mb-2">Try the Demo</h1>
        <p className="text-[#999] mb-2">You are Alex, a 5th grade student.</p>
        <p className="text-[#999] text-sm mb-6 max-w-md text-center">Your tutor has been working with you for months and has built up memories about how you learn. Start a tutoring session.</p>
        
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="text-xs text-[#999] hover:text-[#666] mb-4 flex items-center gap-1 transition-colors"
        >
          <svg className={`w-3 h-3 transition-transform ${showSettings ? 'rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          Settings
        </button>
        
        <AnimatePresence>
          {showSettings && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-6 overflow-hidden"
            >
                <div className="bg-[#fafafa] rounded-lg p-4 space-y-4 text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-[#666]">Memory update frequency</span>
                  <div className="flex items-center gap-3 ml-12">
                    <input
                      type="range"
                      min="1"
                      max="20"
                      value={updateFrequency}
                      onChange={e => setUpdateFrequency(Number(e.target.value))}
                      className="w-24 h-1 bg-[#e5e5e5] rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[#666] [&::-webkit-slider-thumb]:rounded-full"
                    />
                    <span className="w-16 text-[#999] text-right">{updateFrequency} token{updateFrequency > 1 ? 's' : ''}</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[#666]">Max memories in context</span>
                  <div className="flex items-center gap-3 ml-12">
                    <input
                      type="range"
                      min="1"
                      max="15"
                      value={maxMemories}
                      onChange={e => setMaxMemories(Number(e.target.value))}
                      className="w-24 h-1 bg-[#e5e5e5] rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[#666] [&::-webkit-slider-thumb]:rounded-full"
                    />
                    <span className="w-16 text-[#999] text-right">{maxMemories}</span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        
        <div className="w-full max-w-2xl">
          <div className="relative">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendMessage()}
              placeholder="Message..."
              autoFocus
              className="w-full px-4 py-3 pr-12 rounded-full bg-[#f5f5f5] text-[#1a1a1a] placeholder-[#999] focus:outline-none focus:ring-2 focus:ring-[#ddd]"
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim()}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-1.5 rounded-full text-[#999] hover:text-[#666] disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    )
  }

  // Chat view - input at bottom
  return (
    <div className="h-screen overflow-hidden bg-white flex flex-col">
      {/* Back button */}
      <div className="flex-shrink-0 px-4 pt-3">
        <button
          onClick={() => { setShowDemo(false); setMessages([]) }}
          className="text-xs text-[#999] hover:text-[#666] flex items-center gap-1 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back
        </button>
      </div>
      
      {/* Chat History */}
      <div ref={chatRef} className="flex-1 overflow-y-auto px-4">
        <div className="max-w-2xl mx-auto py-6 space-y-6">
          {messages.map((msg, i) => {
            const isLastUserMessage = msg.role === 'user' && i === messages.length - 2
            return (
              <div key={i} ref={isLastUserMessage ? lastMessageRef : null}>
                {/* User message */}
                {msg.role === 'user' && (
                  <div className="flex justify-end mb-4">
                    <div className="bg-[#f5f5f5] rounded-2xl px-4 py-3 max-w-[80%]">
                      <p className="text-[#1a1a1a]">{msg.content}</p>
                    </div>
                  </div>
                )}

                {/* Assistant message */}
                {msg.role === 'assistant' && (
                  <div className="mb-4">
                    {/* Thinking */}
                    {(msg.thinkingTimeline?.length > 0 || (msg.streaming && thinking)) && (
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
                        {msg.timeline?.length > 0 ? (
                          renderTokenizedText(msg.timeline)
                        ) : (
                          msg.content
                        )}
                        {msg.streaming && <span className="cursor" />}
                      </p>
                    </div>

                    {/* Active Memories */}
                    {msg.streaming && currentMemories.length > 0 && (
                      <div className="mt-4 pl-1">
                        <div className="text-xs text-[#999] mb-2 flex items-center gap-2">
                          <span>Active memories</span>
                          {memoryUpdates > 0 && (
                            <span className="text-[#bbb]">({memoryUpdates} updates)</span>
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
            )
          })}
        </div>
      </div>

      {/* Input Area */}
      <div className="flex-shrink-0 bg-white px-4 py-4">
        <div className="max-w-2xl mx-auto">
          <div className="flex justify-center mb-2">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="text-xs text-[#999] hover:text-[#666] flex items-center gap-1 transition-colors"
            >
              <svg className={`w-3 h-3 transition-transform ${showSettings ? 'rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
              Settings
            </button>
          </div>
          
          <AnimatePresence>
            {showSettings && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-3 overflow-hidden"
              >
                <div className="bg-[#fafafa] rounded-lg p-3 space-y-3 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="text-[#666]">Memory update frequency</span>
                    <div className="flex items-center gap-3 ml-12">
                      <input
                        type="range"
                        min="1"
                        max="20"
                        value={updateFrequency}
                        onChange={e => setUpdateFrequency(Number(e.target.value))}
                        disabled={isStreaming}
                        className="w-24 h-1 bg-[#e5e5e5] rounded-full appearance-none cursor-pointer disabled:opacity-50 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[#666] [&::-webkit-slider-thumb]:rounded-full"
                      />
                      <span className="w-16 text-[#999] text-right">{updateFrequency} token{updateFrequency > 1 ? 's' : ''}</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-[#666]">Max memories in context</span>
                    <div className="flex items-center gap-3 ml-12">
                      <input
                        type="range"
                        min="1"
                        max="15"
                        value={maxMemories}
                        onChange={e => setMaxMemories(Number(e.target.value))}
                        disabled={isStreaming}
                        className="w-24 h-1 bg-[#e5e5e5] rounded-full appearance-none cursor-pointer disabled:opacity-50 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[#666] [&::-webkit-slider-thumb]:rounded-full"
                      />
                      <span className="w-16 text-[#999] text-right">{maxMemories}</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          
          <div className="relative">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendMessage()}
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
                <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
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
            <div className="text-xs text-[#999] mb-2">Memories at this token:</div>
            <div className="space-y-1.5 max-h-48 overflow-y-auto">
              {hoveredMemories.map((mem, i) => (
                <div key={i} className="text-xs text-[#666] bg-[#f5f5f5] rounded px-2 py-1.5">
                  {mem}
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
