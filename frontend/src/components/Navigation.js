import React from 'react';
import { Link } from 'react-router-dom';

const Navigation = () => {
  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
      <div className="container">
        <Link className="navbar-brand" to="/">Car Damage Detection</Link>
        <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
          <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse" id="navbarNav">
          <ul className="navbar-nav">
            <li className="nav-item">
              <Link className="nav-link" to="/">Dent Detection</Link>
            </li>
            <li className="nav-item">
              <Link className="nav-link" to="/repair-shops">Find Repair Shops</Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default Navigation; 