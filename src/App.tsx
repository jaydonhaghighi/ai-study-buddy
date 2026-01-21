import { useState, useEffect } from 'react';
import { auth } from './firebase-config';
import { User, onAuthStateChanged } from 'firebase/auth';
import Chat from './components/Chat';
import DataCollector from './components/DataCollector';
import './App.css';

function App() {
  const isCollector = typeof window !== 'undefined' && window.location.pathname === '/collect';
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (isCollector) {
      setLoading(false);
      return;
    }
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setLoading(false);
    });

    return () => unsubscribe();
  }, [isCollector]);

  if (loading) {
    return (
      <div className="container">
        <div className="loading">Loading...</div>
      </div>
    );
  }

  if (isCollector) {
    return <DataCollector />;
  }

  return (
    <div className="app-container">
      <Chat user={user} />
    </div>
  );
}

export default App;

