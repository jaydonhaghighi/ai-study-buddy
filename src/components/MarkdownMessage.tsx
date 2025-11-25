import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';
import './MarkdownMessage.css';

interface MarkdownMessageProps {
  content: string;
  isAI?: boolean;
}

export default function MarkdownMessage({ content, isAI = false }: MarkdownMessageProps) {
  return (
    <div className={`markdown-message ${isAI ? 'markdown-ai' : 'markdown-user'}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ node, inline, className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || '');
            const codeString = String(children).replace(/\n$/, '');
            
            return !inline && match ? (
              <SyntaxHighlighter
                style={vscDarkPlus}
                language={match[1]}
                PreTag="div"
                className="code-block"
                {...props}
              >
                {codeString}
              </SyntaxHighlighter>
            ) : (
              <code className="inline-code" {...props}>
                {children}
              </code>
            );
          },
          p({ children }) {
            return <p className="markdown-paragraph">{children}</p>;
          },
          h1({ children }) {
            return <h1 className="markdown-heading markdown-h1">{children}</h1>;
          },
          h2({ children }) {
            return <h2 className="markdown-heading markdown-h2">{children}</h2>;
          },
          h3({ children }) {
            return <h3 className="markdown-heading markdown-h3">{children}</h3>;
          },
          ul({ children }) {
            return <ul className="markdown-list">{children}</ul>;
          },
          ol({ children }) {
            return <ol className="markdown-list markdown-ordered-list">{children}</ol>;
          },
          li({ children }) {
            return <li className="markdown-list-item">{children}</li>;
          },
          blockquote({ children }) {
            return <blockquote className="markdown-blockquote">{children}</blockquote>;
          },
          table({ children }) {
            return <div className="markdown-table-wrapper"><table className="markdown-table">{children}</table></div>;
          },
          a({ href, children }) {
            return <a href={href} target="_blank" rel="noopener noreferrer" className="markdown-link">{children}</a>;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

