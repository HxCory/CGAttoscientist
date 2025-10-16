import { useState, useRef, useEffect } from 'react'
import { Upload, Send, Loader2, BookOpen, Sparkles, FileText } from 'lucide-react'
import axios from 'axios'
import './App.css'

const API_BASE = '/api'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [stats, setStats] = useState(null)
  const [documentId, setDocumentId] = useState(null)
  const messagesEndRef = useRef(null)
  const fileInputRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    // Fetch initial stats
    fetchStats()
  }, [])

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_BASE}/stats`)
      setStats(response.data)
    } catch (error) {
      console.error('Error fetching stats:', error)
    }
  }

  const handleFileUpload = async (event) => {
    const file = event.target.files[0]
    if (!file) return

    setUploading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      setDocumentId(response.data.document_id)
      
      // Add system message
      setMessages(prev => [...prev, {
        type: 'system',
        content: `✓ Successfully processed thesis: ${response.data.pages_processed} pages, ${response.data.chunks_created} chunks created.`
      }])

      fetchStats()
    } catch (error) {
      console.error('Error uploading file:', error)
      setMessages(prev => [...prev, {
        type: 'error',
        content: `Failed to upload file: ${error.response?.data?.detail || error.message}`
      }])
    } finally {
      setUploading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const question = input.trim()
    setInput('')
    setLoading(true)

    // Add user message
    setMessages(prev => [...prev, {
      type: 'user',
      content: question
    }])

    try {
      // For now, use non-streaming
      const response = await axios.post(`${API_BASE}/ask`, {
        question: question,
        document_id: documentId,
        top_k: 3,
        include_images: true,
        stream: false
      })

      // Add assistant message with sources
      setMessages(prev => [...prev, {
        type: 'assistant',
        content: response.data.answer,
        sources: response.data.sources,
        metadata: response.data.metadata
      }])

    } catch (error) {
      console.error('Error asking question:', error)
      setMessages(prev => [...prev, {
        type: 'error',
        content: `Error: ${error.response?.data?.detail || error.message}`
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary-100 rounded-lg">
                <BookOpen className="h-6 w-6 text-primary-600" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-900">
                  PhD Thesis Research Assistant
                </h1>
                <p className="text-sm text-slate-600">
                  Attosecond Streaking & Time Delays in Photoionization
                </p>
              </div>
            </div>

            {/* Upload Button */}
            <div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                onChange={handleFileUpload}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
                className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {uploading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Upload className="h-4 w-4" />
                    Upload Thesis PDF
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Stats Bar */}
          {stats && (
            <div className="mt-4 flex items-center gap-4 text-sm text-slate-600">
              <div className="flex items-center gap-1">
                <FileText className="h-4 w-4" />
                <span>{stats.vector_store.total_chunks} chunks indexed</span>
              </div>
              <div className="flex items-center gap-1">
                <Sparkles className="h-4 w-4" />
                <span>Model: {stats.config.embedding_model}</span>
              </div>
            </div>
          )}
        </div>
      </header>

      {/* Chat Container */}
      <main className="max-w-5xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        <div className="bg-white rounded-2xl shadow-xl border border-slate-200 flex flex-col h-[calc(100vh-250px)]">
          
          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {messages.length === 0 && (
              <div className="text-center py-12">
                <div className="inline-block p-4 bg-primary-50 rounded-full mb-4">
                  <BookOpen className="h-12 w-12 text-primary-600" />
                </div>
                <h2 className="text-xl font-semibold text-slate-900 mb-2">
                  Welcome to Your Research Assistant
                </h2>
                <p className="text-slate-600 max-w-md mx-auto">
                  Upload your PhD thesis PDF to get started, then ask questions about your research.
                  I can help you remember details, connect concepts, and explore new directions.
                </p>
              </div>
            )}

            {messages.map((message, index) => (
              <Message key={index} message={message} />
            ))}

            {loading && (
              <div className="flex items-center gap-2 text-slate-600">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Thinking...</span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="border-t border-slate-200 p-4">
            <form onSubmit={handleSubmit} className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a question about your research..."
                disabled={loading}
                className="flex-1 px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <button
                type="submit"
                disabled={loading || !input.trim()}
                className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                <Send className="h-4 w-4" />
                Send
              </button>
            </form>
          </div>
        </div>
      </main>
    </div>
  )
}

function Message({ message }) {
  if (message.type === 'system') {
    return (
      <div className="flex justify-center">
        <div className="px-4 py-2 bg-blue-50 text-blue-700 rounded-lg text-sm border border-blue-200">
          {message.content}
        </div>
      </div>
    )
  }

  if (message.type === 'error') {
    return (
      <div className="flex justify-center">
        <div className="px-4 py-2 bg-red-50 text-red-700 rounded-lg text-sm border border-red-200">
          {message.content}
        </div>
      </div>
    )
  }

  if (message.type === 'user') {
    return (
      <div className="flex justify-end message-enter">
        <div className="max-w-[80%] px-4 py-3 bg-primary-600 text-white rounded-2xl rounded-tr-sm">
          {message.content}
        </div>
      </div>
    )
  }

  if (message.type === 'assistant') {
    return (
      <div className="flex flex-col gap-2 message-enter">
        <div className="max-w-[85%] px-4 py-3 bg-slate-100 text-slate-900 rounded-2xl rounded-tl-sm">
          <div className="prose prose-sm max-w-none">
            {message.content}
          </div>
        </div>
        
        {/* Sources */}
        {message.sources && message.sources.pages && message.sources.pages.length > 0 && (
          <div className="max-w-[85%] px-4 py-2 bg-slate-50 rounded-lg border border-slate-200">
            <div className="text-xs font-semibold text-slate-600 mb-1">
              📖 Sources:
            </div>
            <div className="flex flex-wrap gap-2">
              {message.sources.pages.map((page) => (
                <span
                  key={page}
                  className="px-2 py-1 bg-white text-slate-700 rounded text-xs border border-slate-200"
                >
                  Page {page}
                </span>
              ))}
            </div>
            {message.metadata && message.metadata.tokens_used && (
              <div className="text-xs text-slate-500 mt-2">
                Tokens: {message.metadata.tokens_used.input} in / {message.metadata.tokens_used.output} out
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  return null
}

export default App

