const GOOGLE_MAPS_API_KEY = process.env.REACT_APP_GOOGLE_MAPS_API_KEY;

export const fetchNearbyShops = async (lat, lng, sortBy = 'rating') => {
  try {
    const response = await fetch(
      `https://maps.googleapis.com/maps/api/place/nearbysearch/json?` +
      `location=${lat},${lng}&` +
      `radius=1500&` +  // 1.5km radius
      `type=car_repair&` +
      `key=${GOOGLE_MAPS_API_KEY}`
    );

    if (!response.ok) {
      throw new Error('Failed to fetch nearby shops');
    }

    const data = await response.json();
    
    // Sort the results
    if (data.results) {
      data.results = sortShops(data.results, sortBy, lat, lng);
    }
    
    return data;
  } catch (error) {
    console.error('Error fetching nearby shops:', error);
    throw error;
  }
};

// Helper function to calculate distance between two points
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

// Sort shops by rating or distance
const sortShops = (shops, sortBy, userLat, userLng) => {
  return [...shops].sort((a, b) => {
    if (sortBy === 'rating') {
      // Sort by rating (highest first)
      const ratingA = a.rating || 0;
      const ratingB = b.rating || 0;
      return ratingB - ratingA;
    } else {
      // Sort by distance (nearest first)
      const distA = calculateDistance(
        userLat, 
        userLng, 
        a.geometry.location.lat, 
        a.geometry.location.lng
      );
      const distB = calculateDistance(
        userLat, 
        userLng, 
        b.geometry.location.lat, 
        b.geometry.location.lng
      );
      return distA - distB;
    }
  });
}; 