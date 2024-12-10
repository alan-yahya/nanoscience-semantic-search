const ToggleButton = ({ isRecursive, onToggle, isLoading }) => {
    return React.createElement(
        'div',
        { className: 'toggle-container' },
        React.createElement(
            'button',
            {
                id: 'toggleEmbeddings',
                className: 'btn btn-outline-secondary btn-sm',
                onClick: onToggle,
                disabled: isLoading
            },
            [
                'Using: ',
                React.createElement(
                    'span',
                    { 
                        id: 'embeddingType',
                        key: 'embeddingType'
                    },
                    isRecursive ? 'Recursive' : 'Standard'
                ),
                ' Embeddings'
            ]
        )
    );
}; 