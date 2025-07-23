import React, { useState, useEffect } from 'react';
import './RepairShopFinder.css';

const RepairShopFinder = () => {
    const [location, setLocation] = useState('');
    const [shops, setShops] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSearch = async (e) => {
        e.preventDefault();
        if (!location.trim()) return;

        setIsLoading(true);
        setError(null);

        try {
            // Simulate API call
            await new Promise(resolve => setTimeout(resolve, 1500));
            setShops([
                {
                    id: 1,
                    name: 'AutoCare Plus',
                    rating: 4.8,
                    reviews: 156,
                    distance: '2.3',
                    address: '123 Main St, City, State',
                    services: ['Dent Repair', 'Paintless Dent Removal', 'Auto Body'],
                    openNow: true,
                    estimatedWait: '1-2 days'
                },
                {
                    id: 2,
                    name: 'Dent Masters',
                    rating: 4.6,
                    reviews: 89,
                    distance: '3.1',
                    address: '456 Oak Ave, City, State',
                    services: ['Paintless Dent Removal', 'Auto Detailing'],
                    openNow: true,
                    estimatedWait: '2-3 days'
                },
                {
                    id: 3,
                    name: 'Quick Fix Auto',
                    rating: 4.4,
                    reviews: 203,
                    distance: '4.5',
                    address: '789 Pine Rd, City, State',
                    services: ['Dent Repair', 'Auto Body', 'Paint'],
                    openNow: false,
                    estimatedWait: '3-4 days'
                }
            ]);
        } catch (err) {
            setError('Failed to fetch repair shops. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="repair-shop-finder">
            <div className="finder-content">
                <h1 className="section-title">Find Repair Shops</h1>
                <p className="section-description">
                    Enter your location to find the nearest auto repair shops specializing in dent removal.
                </p>

                <form onSubmit={handleSearch} className="search-form">
                    <div className="search-input-wrapper">
                        <input
                            type="text"
                            value={location}
                            onChange={(e) => setLocation(e.target.value)}
                            placeholder="Enter your location"
                            className="search-input"
                        />
                        <button 
                            type="submit" 
                            className="search-btn"
                            disabled={isLoading || !location.trim()}
                        >
                            {isLoading ? (
                                <span className="spinner"></span>
                            ) : (
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <circle cx="11" cy="11" r="8" />
                                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                                </svg>
                            )}
                        </button>
                    </div>
                </form>

                {error && (
                    <div className="error-message">
                        {error}
                    </div>
                )}

                <div className="shops-grid">
                    {shops.map((shop, index) => (
                        <div 
                            key={shop.id} 
                            className="shop-card"
                            style={{ animationDelay: `${index * 0.1}s` }}
                        >
                            <div className="shop-header">
                                <h3 className="shop-name">{shop.name}</h3>
                                <div className="shop-rating">
                                    <span className="rating-value">{shop.rating}</span>
                                    <div className="stars">
                                        {[...Array(5)].map((_, i) => (
                                            <span 
                                                key={i} 
                                                className={`star ${i < Math.floor(shop.rating) ? 'filled' : ''}`}
                                            >
                                                ★
                                            </span>
                                        ))}
                                    </div>
                                    <span className="review-count">({shop.reviews} reviews)</span>
                                </div>
                            </div>

                            <div className="shop-details">
                                <div className="detail-item">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
                                        <circle cx="12" cy="10" r="3" />
                                    </svg>
                                    <span>{shop.distance} miles away</span>
                                </div>
                                <div className="detail-item">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z" />
                                        <circle cx="12" cy="10" r="3" />
                                    </svg>
                                    <span>{shop.address}</span>
                                </div>
                                <div className="detail-item">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                        <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
                                        <line x1="16" y1="2" x2="16" y2="6" />
                                        <line x1="8" y1="2" x2="8" y2="6" />
                                        <line x1="3" y1="10" x2="21" y2="10" />
                                    </svg>
                                    <span>Wait time: {shop.estimatedWait}</span>
                                </div>
                            </div>

                            <div className="shop-services">
                                {shop.services.map((service, i) => (
                                    <span key={i} className="service-tag">{service}</span>
                                ))}
                            </div>

                            <div className="shop-footer">
                                <span className={`status ${shop.openNow ? 'open' : 'closed'}`}>
                                    {shop.openNow ? 'Open Now' : 'Closed'}
                                </span>
                                <button className="book-btn">Book Appointment</button>
                            </div>
                        </div>
                    ))}
                </div>

                {shops.length === 0 && !isLoading && !error && (
                    <div className="no-results">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="12" r="10" />
                            <line x1="12" y1="8" x2="12" y2="12" />
                            <line x1="12" y1="16" x2="12.01" y2="16" />
                        </svg>
                        <p>Enter your location to find repair shops near you</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default RepairShopFinder; 