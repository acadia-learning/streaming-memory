import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

// Animated memory visualization
function MemoryVisualization({ isStreaming = true }) {
  const [activeMemories, setActiveMemories] = useState([0, 1, 2])
  const [tokenIndex, setTokenIndex] = useState(0)
  
  const memories = [
    "Loves dinosaurs, especially T-Rex",
    "Gets frustrated with decimal placement",
    "Benefits from step-by-step guidance",
    "Prefers morning sessions",
    "Responds well to humor",
    "Needs movement breaks every 20 min",
  ]
  
  const tokens = ["Let", " me", " help", " you", " with", " that", " fraction", " problem", "..."]
  
  useEffect(() => {
    if (!isStreaming) return
    const interval = setInterval(() => {
      setTokenIndex(prev => (prev + 1) % tokens.length)
      // Shift memories at certain tokens
      if (tokenIndex === 3 || tokenIndex === 6) {
        setActiveMemories(prev => {
          const next = [...prev]
          next.shift()
          next.push((prev[2] + 1) % memories.length)
          return next
        })
      }
    }, 400)
    return () => clearInterval(interval)
  }, [isStreaming, tokenIndex])
  
  return (
    <div className="bg-[#fafafa] rounded-2xl p-6 max-w-lg mx-auto">
      {/* Token stream */}
      <div className="mb-6">
        <div className="text-xs text-[#999] mb-2 font-medium">Generated response</div>
        <div className="bg-white rounded-lg p-4 min-h-[60px] font-mono text-sm">
          {tokens.slice(0, tokenIndex + 1).map((token, i) => (
            <span 
              key={i} 
              className={i === tokenIndex ? 'bg-blue-100 text-blue-700' : 'text-[#333]'}
            >
              {token}
            </span>
          ))}
          <span className="inline-block w-0.5 h-4 bg-blue-500 animate-pulse ml-0.5" />
        </div>
      </div>
      
      {/* Active memories */}
      <div>
        <div className="text-xs text-[#999] mb-2 font-medium flex items-center gap-2">
          Active memories
          {isStreaming && (
            <span className="text-green-600 text-[10px]">● updating live</span>
          )}
        </div>
        <div className="space-y-2">
          {activeMemories.map((memIdx, i) => (
            <motion.div
              key={`${memIdx}-${i}`}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="bg-white rounded-lg px-3 py-2 text-xs text-[#666] border-l-2 border-blue-400"
            >
              {memories[memIdx]}
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  )
}

// Comparison diagram
function ComparisonDiagram() {
  return (
    <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
      {/* Traditional RAG */}
      <div className="text-center">
        <h3 className="text-lg font-semibold text-[#666] mb-4">Traditional RAG</h3>
        <div className="bg-[#fafafa] rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-center gap-2">
            <div className="w-20 h-10 bg-blue-100 rounded flex items-center justify-center text-xs text-blue-700">Query</div>
            <svg className="w-6 h-6 text-[#ccc]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
            <div className="w-20 h-10 bg-purple-100 rounded flex items-center justify-center text-xs text-purple-700">Retrieve</div>
            <svg className="w-6 h-6 text-[#ccc]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
            <div className="w-20 h-10 bg-green-100 rounded flex items-center justify-center text-xs text-green-700">Generate</div>
          </div>
          <div className="text-xs text-[#999] pt-2 border-t border-[#eee]">
            Context is <span className="text-red-500 font-medium">fixed</span> after initial retrieval
          </div>
        </div>
      </div>
      
      {/* Streaming Memory */}
      <div className="text-center">
        <h3 className="text-lg font-semibold text-[#333] mb-4">Streaming Memory</h3>
        <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-2xl p-6 space-y-4 ring-2 ring-blue-200">
          <div className="flex items-center justify-center gap-2">
            <div className="w-20 h-10 bg-blue-100 rounded flex items-center justify-center text-xs text-blue-700">Query</div>
            <svg className="w-6 h-6 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
            <div className="relative">
              <div className="w-20 h-10 bg-purple-100 rounded flex items-center justify-center text-xs text-purple-700">Retrieve</div>
              <svg className="absolute -bottom-6 left-1/2 -translate-x-1/2 w-4 h-4 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
              </svg>
            </div>
            <svg className="w-6 h-6 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
            <div className="w-20 h-10 bg-green-100 rounded flex items-center justify-center text-xs text-green-700">Generate</div>
          </div>
          <div className="flex items-center justify-center">
            <svg className="w-32 h-8 text-purple-300" viewBox="0 0 120 30">
              <path d="M110 5 Q 60 25, 10 5" fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray="4 2">
                <animate attributeName="stroke-dashoffset" from="0" to="12" dur="1s" repeatCount="indefinite" />
              </path>
              <polygon points="10,2 10,8 4,5" fill="currentColor" />
            </svg>
          </div>
          <div className="text-xs text-[#666] pt-2 border-t border-blue-100">
            Context <span className="text-green-600 font-medium">adapts</span> as the model reasons
          </div>
        </div>
      </div>
    </div>
  )
}

// Hebbian visualization
function HebbianVisualization() {
  const [strength, setStrength] = useState(0.3)
  
  useEffect(() => {
    const interval = setInterval(() => {
      setStrength(prev => prev < 0.9 ? prev + 0.1 : 0.3)
    }, 800)
    return () => clearInterval(interval)
  }, [])
  
  return (
    <div className="bg-[#fafafa] rounded-2xl p-6 max-w-md mx-auto">
      <div className="flex items-center justify-center gap-8">
        <div className="text-center">
          <div className="w-16 h-16 rounded-full bg-blue-100 flex items-center justify-center text-2xl mb-2">🦕</div>
          <div className="text-xs text-[#666]">dinosaurs</div>
        </div>
        
        <div className="relative">
          <svg className="w-24 h-8" viewBox="0 0 100 30">
            <line 
              x1="0" y1="15" x2="100" y2="15" 
              stroke={`rgba(59, 130, 246, ${strength})`}
              strokeWidth={Math.max(1, strength * 6)}
              strokeLinecap="round"
            />
          </svg>
          <div className="absolute -bottom-4 left-1/2 -translate-x-1/2 text-[10px] text-blue-500">
            {Math.round(strength * 100)}%
          </div>
        </div>
        
        <div className="text-center">
          <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center text-2xl mb-2">🔬</div>
          <div className="text-xs text-[#666]">science</div>
        </div>
      </div>
      
      <div className="mt-6 text-center text-xs text-[#999]">
        Memories retrieved together <span className="text-blue-600">strengthen</span> their connection
      </div>
    </div>
  )
}

export default function Landing({ onStartDemo }) {
  return (
    <div className="min-h-screen bg-white">
      {/* Hero */}
      <section className="px-4 pt-20 pb-16 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="text-4xl md:text-5xl font-bold text-[#1a1a1a] mb-4">
            Streaming Memory
          </h1>
          <p className="text-xl text-[#666] mb-2 max-w-2xl mx-auto">
            Dynamic context that evolves as the model thinks
          </p>
          <p className="text-sm text-[#999] mb-8 max-w-xl mx-auto">
            Re-retrieve memories every token based on the model's reasoning, not just the initial query
          </p>
          
          <button
            onClick={onStartDemo}
            className="px-8 py-3 bg-[#1a1a1a] text-white rounded-full font-medium hover:bg-[#333] transition-colors"
          >
            Try the Demo
          </button>
        </motion.div>
      </section>
      
      {/* Live visualization */}
      <section className="px-4 py-12 bg-gradient-to-b from-white to-[#fafafa]">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <MemoryVisualization />
        </motion.div>
      </section>
      
      {/* Problem/Solution */}
      <section className="px-4 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-2xl font-bold text-center text-[#1a1a1a] mb-4">
            The Problem with Static Context
          </h2>
          <p className="text-center text-[#666] mb-12 max-w-2xl mx-auto">
            Traditional RAG retrieves context once at the start. But as a model reasons through a problem, 
            the relevant context changes. What starts as a question about fractions might lead to 
            insights about emotional state, learning preferences, or past struggles.
          </p>
          
          <ComparisonDiagram />
        </motion.div>
      </section>
      
      {/* How it works */}
      <section className="px-4 py-16 bg-[#fafafa]">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-2xl font-bold text-center text-[#1a1a1a] mb-4">
            How It Works
          </h2>
          <p className="text-center text-[#666] mb-12 max-w-xl mx-auto">
            Every N tokens, we re-embed the recent output and query the memory system again
          </p>
          
          <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
            {[
              {
                step: "1",
                title: "Generate tokens",
                desc: "Model produces output token by token"
              },
              {
                step: "2", 
                title: "Re-embed context",
                desc: "Embed the recent tokens to capture current reasoning"
              },
              {
                step: "3",
                title: "Re-retrieve memories",
                desc: "Query memory with new embedding, swap in relevant context"
              }
            ].map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: i * 0.1 }}
                className="bg-white rounded-xl p-6 text-center"
              >
                <div className="w-10 h-10 rounded-full bg-blue-100 text-blue-700 font-bold flex items-center justify-center mx-auto mb-4">
                  {item.step}
                </div>
                <h3 className="font-semibold text-[#333] mb-2">{item.title}</h3>
                <p className="text-sm text-[#666]">{item.desc}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>
      
      {/* Hebbian Learning */}
      <section className="px-4 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-2xl font-bold text-center text-[#1a1a1a] mb-4">
            Hebbian Reinforcement
          </h2>
          <p className="text-center text-[#666] mb-12 max-w-xl mx-auto">
            "Neurons that fire together, wire together." Memories retrieved together strengthen their association.
          </p>
          
          <HebbianVisualization />
        </motion.div>
      </section>
      
      {/* CTA */}
      <section className="px-4 py-16 text-center bg-gradient-to-b from-[#fafafa] to-white">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-2xl font-bold text-[#1a1a1a] mb-4">
            See it in action
          </h2>
          <p className="text-[#666] mb-8 max-w-md mx-auto">
            Chat with an AI tutor who has months of memories about a student. 
            Watch the memories shift as the conversation evolves.
          </p>
          
          <button
            onClick={onStartDemo}
            className="px-8 py-3 bg-[#1a1a1a] text-white rounded-full font-medium hover:bg-[#333] transition-colors"
          >
            Try the Demo
          </button>
          
          <div className="mt-8 flex items-center justify-center gap-6 text-xs text-[#999]">
            <a href="https://github.com/acadia-learning/streaming-memory" target="_blank" rel="noopener" className="hover:text-[#666] transition-colors">
              GitHub
            </a>
            <span>•</span>
            <span>Built with Qwen3-8B on Modal</span>
          </div>
        </motion.div>
      </section>
    </div>
  )
}

