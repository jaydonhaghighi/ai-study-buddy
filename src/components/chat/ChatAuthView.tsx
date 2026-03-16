import type { FormEvent } from 'react';

type ChatAuthViewProps = {
  showAuth: boolean;
  authEmail: string;
  authPassword: string;
  isSignUp: boolean;
  authError: string | null;
  authLoading: boolean;
  onShowAuth: (value: boolean) => void;
  onEmailChange: (value: string) => void;
  onPasswordChange: (value: string) => void;
  onToggleSignUp: () => void;
  onSubmit: (e: FormEvent) => void;
  onGoogleSignIn: () => Promise<void>;
};

export default function ChatAuthView({
  showAuth,
  authEmail,
  authPassword,
  isSignUp,
  authError,
  authLoading,
  onShowAuth,
  onEmailChange,
  onPasswordChange,
  onToggleSignUp,
  onSubmit,
  onGoogleSignIn,
}: ChatAuthViewProps) {
  return (
    <div className="chat-container chat-container-auth">
      <div className="chat-header chat-header-auth">
        <div className="chat-header-inner">
          <div className="chat-header-left">
            <h2 className="chat-header-title">Echelon</h2>
            <p className="chat-header-subtitle">Sign in to start learning.</p>
          </div>
        </div>
      </div>
      {!showAuth ? (
        <div className="chat-placeholder">
          <div className="auth-prompt">
            <p className="auth-prompt-title">Organize your learning with Courses & Sessions</p>
            <p className="auth-prompt-subtitle">Create a focused workspace for every subject, and keep your study chats in context.</p>
            <button className="auth-toggle-button" onClick={() => onShowAuth(true)} disabled={authLoading}>Sign In / Sign Up</button>
            <button
              type="button"
              className="auth-google-button"
              onClick={() => {
                void onGoogleSignIn();
              }}
              disabled={authLoading}
            >
              {authLoading ? 'Loading...' : 'Continue with Google'}
            </button>
          </div>
        </div>
      ) : (
        <div className="chat-auth-form">
          <form onSubmit={onSubmit}>
            <h3>{isSignUp ? 'Create Account' : 'Sign In'}</h3>
            {authError && <div className="auth-error">{authError}</div>}
            <input
              type="email"
              placeholder="Email"
              value={authEmail}
              onChange={(e) => onEmailChange(e.target.value)}
              className="auth-input"
              required
            />
            <input
              type="password"
              placeholder="Password"
              value={authPassword}
              onChange={(e) => onPasswordChange(e.target.value)}
              className="auth-input"
              required
            />
            <button type="submit" className="auth-submit-button" disabled={authLoading}>
              {authLoading ? 'Loading...' : (isSignUp ? 'Sign Up' : 'Sign In')}
            </button>
            <div className="auth-divider">or</div>
            <button
              type="button"
              className="auth-google-button"
              onClick={() => {
                void onGoogleSignIn();
              }}
              disabled={authLoading}
            >
              {authLoading ? 'Loading...' : 'Continue with Google'}
            </button>
            <button type="button" className="auth-switch-button" onClick={onToggleSignUp} disabled={authLoading}>
              {isSignUp ? 'Have account? Sign In' : 'No account? Sign Up'}
            </button>
            <button type="button" className="auth-cancel-button" onClick={() => onShowAuth(false)} disabled={authLoading}>
              Cancel
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
