const FAQPage = () => {
    const faqs = [
        {
            question: "What is NanoSearch?",
            answer: "NanoSearch is a specialised search engine for nanoscience papers. It uses a pre-trained language model to parse all of the text in every nanoscience paper in the repository."
        },
        {
            question: "What does it use?",
            answer: `The database uses custom embeddings from <a href="https://huggingface.co/Flamenco43/NanoBERT-V2" target="_blank" style="color: #3498db; text-decoration: none;">NanoBERT</a> to resolve queries. It leverages the <a href="https://huggingface.co/docs/api-inference/en/index" target="_blank" style="color: #3498db; text-decoration: none;">Hugging Face Inference API</a> and <a href="https://github.com/facebookresearch/faiss" target="_blank" style="color: #3498db; text-decoration: none;">FAISS</a> to carry out a hybrid sparse and dense vector search. The OpenAI embeddings use <a href="https://platform.openai.com/docs/models/text-embedding-3" target="_blank" style="color: #3498db; text-decoration: none;">text-embedding-3-large</a>.`
        },
        {
            question: "How accurate are the search results?",
            answer: "The search results are ranked by semantic similarity, meaning they capture the meaning of your query, rather than just the keywords. The distance score shown with each result indicates how closely it matches your query - lower scores indicate better matches."
        },
        {
            question: "What do the FAISS vector counts mean?",
            answer: "The number of FAISS vectors reflects the segmentation strategy, which is influenced by the context window of each model (512 tokens for BERT models, 8191 tokens for OpenAI embedding models)."
        },
        {
            question: "Can I access the full papers?",
            answer: "Nanosearch provides titles and DOIs (Digital Object Identifiers) for the papers. You can use these DOIs to access the full papers through the original publishers."
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
                            React.createElement('p', { 
                                className: 'text-muted', 
                                key: 'a',
                                dangerouslySetInnerHTML: { __html: faq.answer }
                            })
                        ]
                    )
                )
            )
        ]
    );
}; 