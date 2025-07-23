import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { FaMapMarkerAlt, FaStar, FaPhoneAlt, FaClock } from 'react-icons/fa';
import axios from 'axios';

// Fix for default marker icons in Leaflet with React
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const shopIcon = new L.Icon({
  iconUrl: 'https://cdn-icons-png.flaticon.com/512/684/684908.png',
  iconSize: [32, 32],
  iconAnchor: [16, 32],
  popupAnchor: [0, -32],
});

const userIcon = new L.Icon({
  iconUrl: 'https://cdn-icons-png.flaticon.com/512/149/149060.png',
  iconSize: [32, 32],
  iconAnchor: [16, 32],
  popupAnchor: [0, -32],
});

const defaultLocation = { lat: 19.133, lon: 72.822 };
const GOOGLE_MAPS_API_KEY = process.env.REACT_APP_GOOGLE_MAPS_API_KEY;

const RepairShopFinder = () => {
  const [shops, setShops] = useState([]);
  const [selectedShop, setSelectedShop] = useState(null);
  const [sortBy, setSortBy] = useState('distance');
  const [userLocation, setUserLocation] = useState(defaultLocation);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchNearbyShops = async (latitude, longitude, sortType = sortBy) => {
    try {
      const response = await axios.get(
        `http://127.0.0.1:8000/google-places-nearby?lat=${latitude}&lon=${longitude}&sort_by=${sortType}`
      );

      if (response.data.results) {
        const shopsWithDetails = await Promise.all(
          response.data.results.map(async (shop) => {
            try {
              // Get additional details for each place
              const detailsResponse = await axios.get(
                `http://127.0.0.1:8000/google-place-details?place_id=${shop.place_id}`
              );

              return {
                ...shop,
                phone: detailsResponse.data.result?.formatted_phone_number || 'N/A',
                opening_hours: {
                  open_now: detailsResponse.data.result?.opening_hours?.open_now || false
                }
              };
            } catch (err) {
              console.error('Error fetching shop details:', err);
              return {
                ...shop,
                phone: 'N/A',
                opening_hours: { open_now: false }
              };
            }
          })
        );
        setShops(shopsWithDetails);
      } else {
        throw new Error('No repair shops found');
      }
    } catch (err) {
      console.error('Error fetching shops:', err);
      setError('Failed to fetch repair shops. Please try again later.');
      setShops([]);
    }
  };

  useEffect(() => {
    const getUserLocation = () => {
      if (navigator.geolocation) {
        console.log('Requesting user location...');
        navigator.geolocation.getCurrentPosition(
          async (position) => {
            const { latitude, longitude } = position.coords;
            console.log('Got user location:', { latitude, longitude });
            setUserLocation({ lat: latitude, lon: longitude });
            await fetchNearbyShops(latitude, longitude, sortBy);
            setLoading(false);
          },
          (error) => {
            console.error('Geolocation error:', error);
            let errorMessage = 'Could not get your location. ';
            switch(error.code) {
              case error.PERMISSION_DENIED:
                errorMessage += 'Please allow location access in your browser settings.';
                break;
              case error.POSITION_UNAVAILABLE:
                errorMessage += 'Location information is unavailable.';
                break;
              case error.TIMEOUT:
                errorMessage += 'Location request timed out.';
                break;
              default:
                errorMessage += 'An unknown error occurred.';
            }
            setError(errorMessage);
            setUserLocation(defaultLocation);
            setShops([]);
            setLoading(false);
          },
          {
            enableHighAccuracy: true,
            timeout: 5000,
            maximumAge: 0
          }
        );
      } else {
        console.error('Geolocation not supported');
        setError('Geolocation is not supported by your browser.');
        setUserLocation(defaultLocation);
        setShops([]);
        setLoading(false);
      }
    };

    getUserLocation();
    // eslint-disable-next-line
  }, []);

  // Re-fetch shops when sortBy changes
  useEffect(() => {
    if (userLocation && userLocation.lat && userLocation.lon) {
      setLoading(true);
      fetchNearbyShops(userLocation.lat, userLocation.lon, sortBy).then(() => setLoading(false));
    }
    // eslint-disable-next-line
  }, [sortBy]);

  const calculateDistance = (shop) => {
    // Use distance_km from backend if available
    if (shop.distance_km !== undefined) return shop.distance_km;
    const R = 6371; // Earth's radius in km
    const lat1 = userLocation.lat * Math.PI / 180;
    const lat2 = shop.geometry.location.lat * Math.PI / 180;
    const deltaLat = (shop.geometry.location.lat - userLocation.lat) * Math.PI / 180;
    const deltaLng = (shop.geometry.location.lng - userLocation.lon) * Math.PI / 180;

    const a = Math.sin(deltaLat/2) * Math.sin(deltaLat/2) +
              Math.cos(lat1) * Math.cos(lat2) *
              Math.sin(deltaLng/2) * Math.sin(deltaLng/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  };

  const sortedShops = shops; // Already sorted by backend

  const handleSortChange = (newSortBy) => {
    setSortBy(newSortBy);
  };

  if (loading) {
    return (
      <div className="container py-4 text-center">
        <div className="spinner-border text-primary" role="status">
          <span className="visually-hidden">Loading...</span>
        </div>
        <p className="mt-2">Finding repair shops near you...</p>
      </div>
    );
  }

  return (
    <div className="container py-4">
      <div className="mb-4 text-center">
        <h1 className="display-5 fw-bold mb-2">Nearby Repair Shops</h1>
        <p className="lead text-muted mb-3">Find trusted auto repair shops near your location. Click a shop for more details!</p>
        {error && <div className="alert alert-warning">{error}</div>}
        <div className="d-flex justify-content-center gap-2 mb-2">
          <button
            className={`btn btn-outline-primary${sortBy === 'distance' ? ' active' : ''}`}
            onClick={() => handleSortChange('distance')}
          >
            Sort by Distance
          </button>
          <button
            className={`btn btn-outline-primary${sortBy === 'rating' ? ' active' : ''}`}
            onClick={() => handleSortChange('rating')}
          >
            Sort by Rating
          </button>
        </div>
      </div>

      <div className="row g-4 align-items-start">
        <div className="col-lg-7">
          <div className="rounded shadow-sm overflow-hidden mb-3" style={{ height: 400 }}>
            <MapContainer
              center={[userLocation.lat, userLocation.lon]}
              zoom={14}
              style={{ height: '100%', width: '100%' }}
              scrollWheelZoom={true}
            >
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution="&copy; OpenStreetMap contributors"
              />
              <Marker position={[userLocation.lat, userLocation.lon]} icon={userIcon}>
                <Popup>You are here</Popup>
              </Marker>
              {sortedShops.slice(0, 5).map(shop => (
                <Marker
                  key={shop.place_id}
                  position={[shop.geometry.location.lat, shop.geometry.location.lng]}
                  icon={shopIcon}
                  eventHandlers={{
                    click: () => setSelectedShop(shop)
                  }}
                >
                  <Popup>
                    <strong>{shop.name}</strong><br />
                    {shop.vicinity}<br />
                    <span className="text-muted">{shop.opening_hours?.open_now ? 'Open Now' : 'Closed'}</span>
                  </Popup>
                </Marker>
              ))}
            </MapContainer>
          </div>
        </div>
        <div className="col-lg-5">
          <div className="list-group">
            {sortedShops.map(shop => (
              <div
                key={shop.place_id}
                className={`list-group-item list-group-item-action mb-3 shadow-sm rounded border-0 p-3${selectedShop && selectedShop.place_id === shop.place_id ? ' bg-primary text-white' : ''}`}
                style={{ cursor: 'pointer', transition: 'box-shadow 0.2s' }}
                onClick={() => setSelectedShop(shop)}
              >
                <div className="d-flex align-items-center mb-2">
                  <FaMapMarkerAlt className="me-2 text-danger" />
                  <h5 className="mb-0 flex-grow-1">{shop.name}</h5>
                  <span className="ms-2">
                    <FaStar className="text-warning mb-1" /> {shop.rating || 'N/A'}
                  </span>
                </div>
                <div className="mb-1 text-muted small">{shop.vicinity}</div>
                <div className="d-flex align-items-center gap-3">
                  <span><FaClock className="me-1" /> {shop.opening_hours?.open_now ? 'Open Now' : 'Closed'}</span>
                  <span><FaPhoneAlt className="me-1" /> {shop.phone || 'N/A'}</span>
                  <span className="badge bg-info text-dark">{shop.distance_km ? shop.distance_km.toFixed(2) : 'N/A'} km</span>
                  <a
                    href={`https://www.google.com/maps/search/?api=1&query=${shop.geometry.location.lat},${shop.geometry.location.lng}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn btn-sm btn-outline-success ms-2"
                  >
                    Go
                  </a>
                </div>
              </div>
            ))}
            {sortedShops.length === 0 && (
              <div className="alert alert-info">No repair shops found nearby.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RepairShopFinder; 