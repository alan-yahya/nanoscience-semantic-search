import Dropdown from './Dropdown';

const ChatContainer = ({ isRecursive, faissCount, vectorDim }) => {
    const [vectorRepresentation, setVectorRepresentation] = React.useState('nanobert');
    const [segmentationStrategy, setSegmentationStrategy] = React.useState('recursive');

    const handleVectorChange = (event) => {
        setVectorRepresentation(event.target.value);
        // Add logic to handle vector representation change
    };

    const handleSegmentationChange = (event) => {
        setSegmentationStrategy(event.target.value);
        // Add logic to handle segmentation strategy change
    };
    const [messages, setMessages] = React.useState([]);
    const chatRef = React.useRef(null);

    // Initial welcome message
    React.useEffect(() => {
        const initialMessages = [
            {
                type: 'assistant',
                content: `
                    <strong>Database Statistics</strong><br><br>
                    Number of FAISS Vectors: ${faissCount}<br>
                    Vector Dimensions: ${vectorDim}<br><br>
                    Please describe the topics or content of the nanoscience papers you want to search for.
                `
            }
        ];

        setMessages(initialMessages);
    }, [faissCount, vectorDim]);

    // Scroll to bottom when messages change
    React.useEffect(() => {
        if (chatRef.current) {
            chatRef.current.scrollTop = chatRef.current.scrollHeight;
        }
    }, [messages]);

    return (
        <div className="chat-outer-container">
            <div className="search-settings">
                <Dropdown
                    label="Vector Representation:"
                    options={[
                        { value: 'nanobert', label: 'NanoBERT' },
                        { value: 'openai', label: 'OpenAI' }
                    ]}
                    selectedValue={vectorRepresentation}
                    onChange={handleVectorChange}
                />
                <Dropdown
                    label="Segmentation Strategy:"
                    options={[
                        { value: 'recursive', label: 'Recursive' },
                        { value: 'standard', label: 'Standard' }
                    ]}
                    selectedValue={segmentationStrategy}
                    onChange={handleSegmentationChange}
                />
            </div>
            <div className="chat-scroll-area" ref={chatRef}>
                <div className="chat-container" id="chatContainer">
                    {messages.map((message, index) => (
                        <div
                            key={index}
                            className={`${message.type}-message`}
                            dangerouslySetInnerHTML={{ __html: message.content }}
                        />
                    ))}
                </div>
            </div>
        </div>
    );
}; 
