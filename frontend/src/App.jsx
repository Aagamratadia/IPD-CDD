import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import DentDetection from './components/DentDetection';
import RepairShopFinder from './components/RepairShopFinder';
import Signup from './components/Signup';
import ChatInterface from './components/ChatInterface';

function App() {
  return (
    <Router>
      <div className="App">
        <Navigation />
        <Routes>
          <Route path="/" element={<DentDetection />} />
          <Route path="/repair-shops" element={<RepairShopFinder />} />
          <Route path="/signup" element={<Signup />} />
        </Routes>
        <ChatInterface />
      </div>
    </Router>
  );
}

export default App;