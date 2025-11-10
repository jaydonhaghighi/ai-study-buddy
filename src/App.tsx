import { useState, useEffect } from 'react';
import { auth } from './firebase-config';
import { User, onAuthStateChanged } from 'firebase/auth';
import Chat from './components/Chat';
import './App.css';

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  if (loading) {
    return (
      <div className="container">
        <div className="loading">Loading...</div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <Chat user={user} />
    </div>
  );
}

export default App;

