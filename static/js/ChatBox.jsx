// Remove import/export syntax and create a global component
const ChatBox = ({ faissCount, vectorDim }) => {
    const [messages, setMessages] = React.useState([]);
    const [inputValue, setInputValue] = React.useState('');
    const chatContainerRef = React.useRef(null);

    React.useEffect(() => {
        // Add initial stats message and a welcome message
        setMessages([{
            type: 'assistant',
            content: `
                <strong>Database Statistics</strong><br>
                Number of FAISS Vectors: ${faissCount}<br>
                Vector Dimensions: ${vectorDim}<br><br>
                How can I help you search through the nanoscience papers?
            `
        }]);
        // Add welcome message
        setMessages(prev => [...prev, {
            type: 'assistant',
            content: 'Welcome back! How can I assist you with your nanoscience paper search today?'
        }]);
    }, [faissCount, vectorDim]);

    React.useEffect(() => {
        // Scroll to bottom when messages change
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [messages]);

    const formatTitle = (title) => {
        title = title.replace(/[\[\]']/g, '');
        return `<a href="https://www.google.com/search?q=${encodeURIComponent(title)}" target="_blank">${title}</a>`;
    };

    const formatDOI = (doi) => {
        doi = doi.replace(/[\[\]']/g, '');
        doi = doi.replace(/^a/, '').replace(/\.html$/, '');
        return doi.replace(/_/g, '/');
    };

    const sendMessage = async () => {
        if (!inputValue.trim()) return;

        // Add user message
        setMessages(prev => [...prev, {
            type: 'user',
            content: inputValue
        }]);

        try {
            const response = await fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: inputValue,
                    top_k: 5
                })
            });

            const data = await response.json();

            if (data.error) {
                setMessages(prev => [...prev, {
                    type: 'assistant',
                    content: `Error: ${data.error}`
                }]);
            } else {
                const results = data.results
                    .filter(result => result.title && result.doi)
                    .map(result => `
                        Title: ${formatTitle(result.title)}<br>
                        DOI: ${formatDOI(result.doi)}<br>
                        Distance: ${result.distance.toFixed(4)}<br><br>
                    `).join('');

                setMessages(prev => [...prev, {
                    type: 'assistant',
                    content: results || 'No results found'
                }]);
            }
        } catch (error) {
            console.error('Error:', error);
            setMessages(prev => [...prev, {
                type: 'assistant',
                content: 'Error: Could not complete the search'
            }]);
        }

        setInputValue('');
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    };

    return (
        <div className="d-flex flex-column vh-100">
            <div className="flex-grow-1 overflow-auto px-3">
                <div className="chat-container container-fluid" ref={chatContainerRef}>
                    {messages.map((message, index) => (
                        <div
                            key={index}
                            className={`${message.type}-message`}
                            dangerouslySetInnerHTML={{ __html: message.content }}
                        />
                    ))}
                </div>
            </div>
            
            <div className="fixed-bottom bg-white border-top p-3" style={{ bottom: 0 }}>
                <div className="container-fluid">
                    <div className="input-group">
                        <input
                            type="text"
                            className="form-control"
                            placeholder="Type your message..."
                            value={inputValue}
                            onChange={(e) => setInputValue(e.target.value)}
                            onKeyPress={handleKeyPress}
                        />
                        <button 
                            className="btn btn-primary"
                            onClick={sendMessage}
                        >
                            Send
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ChatBox; 
