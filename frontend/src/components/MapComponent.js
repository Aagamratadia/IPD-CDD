import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { mockShops } from '../data/mockShops';

// Fix for default marker icons in Leaflet with React
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const MapComponent = () => {
  const [shops, setShops] = useState([]);
  const [selectedShop, setSelectedShop] = useState(null);
  const [sortBy, setSortBy] = useState('distance');
  const [userLocation, setUserLocation] = useState({
    lat: 19.1345,
    lng: 72.8340
  });

  useEffect(() => {
    // Simulate getting user's location
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setUserLocation({
          lat: position.coords.latitude,
          lng: position.coords.longitude
        });
      },
      (error) => {
        console.log("Error getting location:", error);
      }
    );

    // Use mock data
    setShops(mockShops);
  }, []);

  const calculateDistance = (shop) => {
    const R = 6371; // Earth's radius in km
    const lat1 = userLocation.lat * Math.PI / 180;
    const lat2 = shop.geometry.location.lat * Math.PI / 180;
    const deltaLat = (shop.geometry.location.lat - userLocation.lat) * Math.PI / 180;
    const deltaLng = (shop.geometry.location.lng - userLocation.lng) * Math.PI / 180;

    const a = Math.sin(deltaLat/2) * Math.sin(deltaLat/2) +
              Math.cos(lat1) * Math.cos(lat2) *
              Math.sin(deltaLng/2) * Math.sin(deltaLng/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  };

  const sortedShops = [...shops].sort((a, b) => {
    if (sortBy === 'distance') {
      return calculateDistance(a) - calculateDistance(b);
    } else {
      return b.rating - a.rating;
    }
  });

  return (
    <div className="map-container">
      <div className="sort-controls mb-3">
        <button 
          className={`btn ${sortBy === 'distance' ? 'btn-primary' : 'btn-outline-primary'} me-2`}
          onClick={() => setSortBy('distance')}
        >
          Sort by Distance
        </button>
        <button 
          className={`btn ${sortBy === 'rating' ? 'btn-primary' : 'btn-outline-primary'}`}
          onClick={() => setSortBy('rating')}
        >
          Sort by Rating
        </button>
      </div>

      <div style={{ height: '400px', width: '100%' }}>
        <MapContainer
          center={[userLocation.lat, userLocation.lng]}
          zoom={14}
          style={{ height: '100%', width: '100%' }}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          
          {/* User location marker */}
          <Marker position={[userLocation.lat, userLocation.lng]}>
            <Popup>
              <div>
                <h6>Your Location</h6>
              </div>
            </Popup>
          </Marker>

          {/* Shop markers */}
          {sortedShops.map((shop, index) => (
            <Marker
              key={index}
              position={[shop.geometry.location.lat, shop.geometry.location.lng]}
              eventHandlers={{
                click: () => setSelectedShop(shop)
              }}
            >
              <Popup>
                <div>
                  <h6>{shop.name}</h6>
                  <p>{shop.vicinity}</p>
                  <p>Rating: {shop.rating} ⭐</p>
                  <p>Status: {shop.opening_hours.open_now ? 'Open' : 'Closed'}</p>
                  <p>Distance: {calculateDistance(shop).toFixed(1)} km</p>
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      </div>

      <div className="shop-list mt-3">
        <h5>Nearby Repair Shops</h5>
        <div className="list-group">
          {sortedShops.map((shop, index) => (
            <div 
              key={index} 
              className="list-group-item list-group-item-action"
              onClick={() => setSelectedShop(shop)}
            >
              <div className="d-flex w-100 justify-content-between">
                <h6 className="mb-1">{shop.name}</h6>
                <small>{shop.rating} ⭐</small>
              </div>
              <p className="mb-1">{shop.vicinity}</p>
              <small>
                {calculateDistance(shop).toFixed(1)} km away • 
                {shop.opening_hours.open_now ? ' Open' : ' Closed'}
              </small>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MapComponent; 