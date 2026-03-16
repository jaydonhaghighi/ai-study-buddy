import { useState, useEffect } from 'react';
import { auth } from './firebase-config';
import { User, onAuthStateChanged } from 'firebase/auth';
import Chat from './components/Chat';
import DataCollector from './components/DataCollector';
import logo from './public/logo.png';
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

  let content: JSX.Element;
  if (loading) {
    content = (
      <div className="container">
        <div className="loading">Loading...</div>
      </div>
    );
  } else if (isCollector) {
    content = <DataCollector />;
  } else {
    content = (
      <div className="app-container">
        <Chat user={user} />
      </div>
    );
  }

  return (
    <>
      <div className="app-brandmark" aria-label="Echelon logo">
        <img src={logo} alt="Echelon logo" className="app-brandmark-logo" />
        <span className="app-brandmark-text">Echelon</span>
      </div>
      {content}
    </>
  );
}

export default App;
