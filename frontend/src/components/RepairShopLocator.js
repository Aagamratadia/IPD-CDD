import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import MapComponent from './MapComponent';

const DEFAULT_LOCATION = {
    lat: 19.1340,
    lon: 72.8336
};

// Fallback repair shops data for Mumbai area
const FALLBACK_SHOPS = [
    {
        name: "Auto Service Center",
        rating: 4.5,
        geometry: { location: { lat: 19.1340, lng: 72.8336 } },
        distance_km: 0.5
    },
    {
        name: "Car Care Solutions",
        rating: 4.3,
        geometry: { location: { lat: 19.1440, lng: 72.8436 } },
        distance_km: 1.2
    },
    {
        name: "Premium Auto Repair",
        rating: 4.7,
        geometry: { location: { lat: 19.1240, lng: 72.8236 } },
        distance_km: 1.5
    },
    {
        name: "Quick Fix Garage",
        rating: 4.2,
        geometry: { location: { lat: 19.1540, lng: 72.8536 } },
        distance_km: 2.0
    },
    {
        name: "Expert Auto Service",
        rating: 4.6,
        geometry: { location: { lat: 19.1140, lng: 72.8136 } },
        distance_km: 2.5
    }
];

const RepairShopLocator = () => {
    const [userLocation, setUserLocation] = useState(null);
    const [repairShops, setRepairShops] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [sortBy, setSortBy] = useState('rating');
    const [usingFallback, setUsingFallback] = useState(false);

    const getNearbyRepairShops = useCallback(async (lat, lon) => {
        try {
            const response = await axios.get(
                `https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=${lat},${lon}&radius=5000&type=car_repair&key=${process.env.REACT_APP_GOOGLE_MAPS_API_KEY}`
            );
            
            if (response.data.results && response.data.results.length > 0) {
                const shops = response.data.results.map(shop => ({
                    ...shop,
                    distance_km: calculateDistance(
                        lat,
                        lon,
                        shop.geometry.location.lat,
                        shop.geometry.location.lng
                    )
                }));
                setRepairShops(shops);
                setUsingFallback(false);
            } else {
                throw new Error('No repair shops found');
            }
        } catch (err) {
            console.warn('Failed to fetch repair shops from API, using fallback data:', err);
            const fallbackShops = FALLBACK_SHOPS.map(shop => ({
                ...shop,
                distance_km: calculateDistance(
                    lat,
                    lon,
                    shop.geometry.location.lat,
                    shop.geometry.location.lng
                )
            }));
            setRepairShops(fallbackShops);
            setUsingFallback(true);
            setError('Using sample repair shop data. Real-time data unavailable.');
        } finally {
            setLoading(false);
        }
    }, []);

    const getUserLocation = useCallback(async () => {
        try {
            const response = await axios.get(`https://ipinfo.io/json?token=${process.env.REACT_APP_IPINFO_TOKEN}`);
            const [lat, lon] = response.data.loc.split(',').map(Number);
            setUserLocation({ lat, lon });
            getNearbyRepairShops(lat, lon);
        } catch (err) {
            console.warn('Failed to get user location, using default:', err);
            setError('Using default location. Your location could not be determined.');
            setUserLocation(DEFAULT_LOCATION);
            getNearbyRepairShops(DEFAULT_LOCATION.lat, DEFAULT_LOCATION.lon);
        }
    }, [getNearbyRepairShops]);

    useEffect(() => {
        getUserLocation();
    }, [getUserLocation]);

    const calculateDistance = (lat1, lon1, lat2, lon2) => {
        const R = 6371; // Earth's radius in km
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = 
            Math.sin(dLat/2) * Math.sin(dLat/2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
            Math.sin(dLon/2) * Math.sin(dLon/2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R * c;
    };

    const sortShops = (shops, sortType) => {
        return [...shops].sort((a, b) => {
            if (sortType === 'rating') {
                return (b.rating || 0) - (a.rating || 0);
            } else {
                return a.distance_km - b.distance_km;
            }
        });
    };

    if (loading) {
        return <div className="loading">Loading repair shops...</div>;
    }

    const sortedShops = sortShops(repairShops, sortBy);

    return (
        <div className="repair-shop-locator">
            <div className="controls">
                <h2>Nearby Repair Shops</h2>
                {error && <div className="error-message">{error}</div>}
                <div className="sort-controls">
                    <button 
                        className={sortBy === 'rating' ? 'active' : ''} 
                        onClick={() => setSortBy('rating')}
                    >
                        Sort by Rating
                    </button>
                    <button 
                        className={sortBy === 'distance' ? 'active' : ''} 
                        onClick={() => setSortBy('distance')}
                    >
                        Sort by Distance
                    </button>
                </div>
            </div>

            <div className="map-container">
                {userLocation && (
                    <MapComponent userLocation={userLocation} shops={sortedShops} />
                )}
            </div>

            <div className="shop-list">
                <h3>Top Repair Shops</h3>
                <div className="shop-grid">
                    {sortedShops.slice(0, 5).map((shop, index) => (
                        <div key={index} className="shop-card">
                            <h4>{shop.name}</h4>
                            <p>Rating: {shop.rating || 'N/A'}</p>
                            <p>Distance: {shop.distance_km.toFixed(2)} km</p>
                            {!usingFallback && (
                                <a
                                    href={`https://www.google.com/maps/search/?api=1&query=${shop.geometry.location.lat},${shop.geometry.location.lng}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="directions-button"
                                >
                                    Get Directions
                                </a>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default RepairShopLocator; 