import React, { useState, useRef, useEffect } from 'react';
import './ChatInterface.css';

const ChatInterface = () => {
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isOpen, setIsOpen] = useState(false);
    const [isDarkMode, setIsDarkMode] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Apply theme class to body when dark mode changes
    useEffect(() => {
        document.body.classList.toggle('dark-mode', isDarkMode);
    }, [isDarkMode]);

    const toggleTheme = () => {
        setIsDarkMode(!isDarkMode);
    };

    const formatMessage = (text) => {
        // Split the text into lines
        const lines = text.split('\n');
        
        return lines.map((line, index) => {
            // Handle headings
            if (line.startsWith('#')) {
                return <h3 key={index} className="message-heading">{line.substring(1).trim()}</h3>;
            }
            // Handle bullet points
            else if (line.startsWith('•')) {
                return <div key={index} className="message-bullet">{line}</div>;
            }
            // Handle regular text
            return <div key={index}>{line}</div>;
        });
    };

    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (!inputMessage.trim()) return;

        const userMessage = inputMessage.trim();
        setInputMessage('');
        setMessages(prev => [...prev, { text: userMessage, sender: 'user' }]);
        setIsLoading(true);

        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: userMessage })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            setMessages(prev => [...prev, { text: data.response, sender: 'bot' }]);
        } catch (error) {
            console.error('Error:', error);
            setMessages(prev => [...prev, { 
                text: 'Sorry, I encountered an error. Please try again.', 
                sender: 'bot' 
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const toggleChat = () => {
        setIsOpen(!isOpen);
    };

    return (
        <div className={`chat-widget ${isOpen ? 'open' : ''} ${isDarkMode ? 'dark-mode' : ''}`}>
            {!isOpen && (
                <button className="chat-button" onClick={toggleChat}>
                    <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" strokeWidth="2" fill="none">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                    </svg>
                    <span>Chat with us</span>
                </button>
            )}
            
            <div className="chat-container">
                <div className="chat-header">
                    <h3>Virtual Mechanic Assistant</h3>
                    <div className="header-buttons">
                        <button className="theme-toggle" onClick={toggleTheme}>
                            {isDarkMode ? '☀️' : '🌙'}
                        </button>
                    <button className="close-button" onClick={toggleChat}>×</button>
                    </div>
                </div>
                <div className="chat-messages">
                    {messages.length === 0 && (
                        <div className="welcome-message">
                            <h2>Welcome to Virtual Mechanic Assistant</h2>
                            <p>How can I help you with your vehicle today?</p>
                        </div>
                    )}
                    {messages.map((message, index) => (
                        <div 
                            key={index} 
                            className={`message ${message.sender}-message`}
                        >
                            {message.sender === 'bot' ? formatMessage(message.text) : message.text}
                        </div>
                    ))}
                    {isLoading && (
                        <div className="loading-indicator">
                            <div className="typing-indicator">
                                <span></span>
                                <span></span>
                                <span></span>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>
                <form onSubmit={handleSendMessage} className="chat-input-form">
                    <input
                        type="text"
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        placeholder="Type your message..."
                        disabled={isLoading}
                    />
                    <button type="submit" disabled={isLoading}>
                        Send
                    </button>
                </form>
            </div>
        </div>
    );
};

export default ChatInterface;