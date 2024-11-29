const FAQPage = () => {
    const faqs = [
        {
            question: "What is NanoBERT Search?",
            answer: "NanoBERT Search is a specialized semantic search engine designed for nanoscience papers. It uses custom embeddings from NanoBERT to understand and retrieve relevant scientific papers based on natural language queries."
        },
        {
            question: "How does it work?",
            answer: "The system uses NanoBERT, a BERT model fine-tuned on nanoscience literature, to convert search queries into vector representations. These are then compared with pre-computed paper embeddings using FAISS (Facebook AI Similarity Search) to find the most semantically relevant papers."
        },
        {
            question: "What kind of papers are included?",
            answer: "The database includes peer-reviewed papers from the field of nanoscience and nanotechnology. The papers cover various topics including materials science, quantum physics, and molecular engineering at the nanoscale."
        },
        {
            question: "How accurate are the search results?",
            answer: "The search results are ranked by semantic similarity, meaning they capture the meaning and context of your query rather than just matching keywords. The distance score shown with each result indicates how closely it matches your query - lower scores indicate better matches."
        },
        {
            question: "Can I access the full papers?",
            answer: "The search provides titles and DOIs (Digital Object Identifiers) for the papers. You can use these DOIs to access the full papers through your institutional subscriptions or the publishers' websites."
        }
    ];

    return React.createElement(
        'div',
        { className: 'faq-container' },
        [
            React.createElement('h1', { className: 'mb-4', key: 'title' }, 'Frequently Asked Questions'),
            React.createElement(
                'div',
                { className: 'faq-list', key: 'list' },
                faqs.map((faq, index) => 
                    React.createElement(
                        'div',
                        { key: index, className: 'faq-item mb-4' },
                        [
                            React.createElement('h2', { className: 'h4 mb-3', key: 'q' }, faq.question),
                            React.createElement('p', { className: 'text-muted', key: 'a' }, faq.answer)
                        ]
                    )
                )
            )
        ]
    );
}; 