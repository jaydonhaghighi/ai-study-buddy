import { useState } from 'react';
import { auth } from '../firebase-config';
import { 
  signInWithEmailAndPassword, 
  createUserWithEmailAndPassword,
  signOut,
  User
} from 'firebase/auth';

interface AuthSectionProps {
  user: User | null;
}

export default function AuthSection({ user }: AuthSectionProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSignUp, setIsSignUp] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (isSignUp) {
        await createUserWithEmailAndPassword(auth, email, password);
      } else {
        await signInWithEmailAndPassword(auth, email, password);
      }
      setEmail('');
      setPassword('');
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unknown error occurred');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSignOut = async () => {
    try {
      await signOut(auth);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      }
    }
  };

  if (user) {
    return (
      <section className="card">
        <h2>Authentication</h2>
        <div className="auth-status">
          <p>Signed in as: <strong>{user.email}</strong></p>
        </div>
        <button onClick={handleSignOut} className="btn btn-danger">
          Sign Out
        </button>
      </section>
    );
  }

  return (
    <section className="card">
      <h2>Authentication</h2>
      <div className="auth-status">
        <p>Not signed in</p>
      </div>
      {error && <div className="error-message">{error}</div>}
      <form onSubmit={handleAuth}>
        <div className="input-group">
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <div className="input-group">
          <input
            type="password"
            placeholder="Password (min 6 characters)"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={6}
          />
        </div>
        <div className="auth-buttons">
          <button 
            type="submit" 
            className="btn btn-primary" 
            disabled={loading}
          >
            {loading ? 'Loading...' : isSignUp ? 'Sign Up' : 'Sign In'}
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => setIsSignUp(!isSignUp)}
            disabled={loading}
          >
            {isSignUp ? 'Switch to Sign In' : 'Switch to Sign Up'}
          </button>
        </div>
      </form>
    </section>
  );
}

