export default function BottomControls({ 
  isRecording, 
  onToggleRecording,
  userSpeaking,
  isThinking,
  hasThinking,
  hasResponse,
  isSpeaking 
}) {
  return (
    <div className="flex-shrink-0 px-4 py-4 border-t border-[#eee]">
      <div className="max-w-2xl mx-auto flex items-center justify-center gap-3">
        {/* Status indicator */}
        <div className="flex items-center gap-2 text-xs text-[#999] min-w-[100px]">
          {userSpeaking && (
            <>
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              <span>Listening</span>
            </>
          )}
          {!userSpeaking && isThinking && hasThinking && (
            <>
              <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
              <span>Thinking</span>
            </>
          )}
          {!isThinking && hasResponse && !isSpeaking && (
            <>
              <span className="w-2 h-2 rounded-full bg-green-500" />
              <span className="text-green-600">Done</span>
            </>
          )}
          {isSpeaking && (
            <>
              <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
              <span className="text-blue-600">Speaking</span>
            </>
          )}
        </div>

        {/* Mic button */}
        <button
          onClick={onToggleRecording}
          className={`relative p-4 rounded-full transition-all ${
            isRecording
              ? 'bg-red-500 hover:bg-red-600'
              : 'bg-[#1a1a1a] hover:bg-[#333]'
          }`}
        >
          <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
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
          {isRecording && (
            <div className="absolute inset-0 rounded-full bg-red-500 animate-ping opacity-30" />
          )}
        </button>

        {/* Spacer for symmetry */}
        <div className="min-w-[100px]" />
      </div>
    </div>
  );
}

