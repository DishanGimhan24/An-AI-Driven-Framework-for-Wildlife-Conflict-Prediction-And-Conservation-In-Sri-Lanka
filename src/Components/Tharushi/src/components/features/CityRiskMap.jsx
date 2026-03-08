import 'leaflet/dist/leaflet.css';
import { useEffect, useRef, useState } from 'react';
import { GeoJSON, MapContainer, TileLayer } from 'react-leaflet';

export default function CityRiskMap({ district, city, riskLevel, riskScore }) {
  const [geoJsonData, setGeoJsonData] = useState(null);
  const [cityFeature, setCityFeature] = useState(null);
  const [loading, setLoading] = useState(false);
  const mapRef = useRef(null);

  useEffect(() => {
    if (district && city) {
      loadCityGeoJSON();
    }
  }, [district, city]);

  // Fit map to city bounds whenever cityFeature changes
  useEffect(() => {
    if (cityFeature && mapRef.current) {
      const bounds = calculateBounds(cityFeature);
      if (bounds) {
        mapRef.current.fitBounds(bounds, { padding: [40, 40], animate: true, duration: 0.8 });
      }
    }
  }, [cityFeature]);

  const calculateBounds = (feature) => {
    if (!feature?.geometry) return null;
    try {
      const { type, coordinates } = feature.geometry;
      let allCoords = [];

      if (type === 'MultiPolygon') {
        coordinates.forEach(polygon => polygon.forEach(ring => allCoords.push(...ring)));
      } else if (type === 'Polygon') {
        coordinates.forEach(ring => allCoords.push(...ring));
      }

      if (allCoords.length > 0) {
        const lats = allCoords.map(c => c[1]);
        const lngs = allCoords.map(c => c[0]);
        return [
          [Math.min(...lats), Math.min(...lngs)],
          [Math.max(...lats), Math.max(...lngs)]
        ];
      }
    } catch (error) {
      console.error('Error calculating bounds:', error);
    }
    return null;
  };

  const loadCityGeoJSON = async () => {
    setLoading(true);
    try {
      const response = await fetch('/data/gadm41_LKA_2.json');
      const data = await response.json();

      const districtFeatures = data.features.filter(
        feature => feature.properties.NAME_1 === district
      );

      const selectedCity = districtFeatures.find(
        feature => feature.properties.NAME_2 === city
      );

      if (selectedCity) {
        setGeoJsonData({
          type: 'FeatureCollection',
          features: districtFeatures
        });
        setCityFeature(selectedCity); // triggers the fitBounds effect
      }
    } catch (error) {
      console.error('Failed to load city GeoJSON:', error);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'HIGH': return '#EF4444';
      case 'MEDIUM': return '#F59E0B';
      case 'LOW': return '#10B981';
      default: return '#9CA3AF';
    }
  };

  const getCityStyle = (feature) => {
    if (feature.properties.NAME_2 === city) {
      return {
        fillColor: getRiskColor(riskLevel),
        fillOpacity: 0.8,
        color: '#1F2937',
        weight: 3,
        opacity: 1
      };
    }
    return {
      fillColor: '#E5E7EB',
      fillOpacity: 0.3,
      color: '#9CA3AF',
      weight: 1,
      opacity: 0.5
    };
  };

  const onEachCity = (feature, layer) => {
    if (feature.properties.NAME_2 === city) {
      const popupContent = `
        <div style="padding: 8px;">
          <h3 style="margin: 0 0 8px 0; font-weight: 600; font-size: 14px;">${feature.properties.NAME_2}</h3>
          <div style="margin-bottom: 4px;">
            <span style="font-size: 12px; color: #666;">Risk Level:</span>
            <span style="font-weight: 600; margin-left: 4px; color: ${getRiskColor(riskLevel)};">
              ${riskLevel}
            </span>
          </div>
          <div>
            <span style="font-size: 12px; color: #666;">Risk Score:</span>
            <span style="font-weight: 600; margin-left: 4px;">
              ${Math.round(riskScore * 100)}%
            </span>
          </div>
        </div>
      `;
      layer.bindPopup(popupContent);
      layer.openPopup();
    }
  };

  return (
    <div style={{ position: 'relative', overflow: 'hidden', borderRadius: '12px', height: '384px' }}>
      {/* Loading overlay — MapContainer stays mounted */}
      {loading && (
        <div style={{ position: 'absolute', inset: 0, zIndex: 1001, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(10,14,20,0.7)', backdropFilter: 'blur(4px)', borderRadius: '12px' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ width: '48px', height: '48px', border: '4px solid rgba(255,255,255,0.1)', borderTopColor: 'var(--emerald-500)', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 16px' }} />
            <p style={{ color: '#6b7280', fontSize: '14px' }}>Loading map...</p>
          </div>
        </div>
      )}

      <MapContainer
        ref={mapRef}
        center={[7.8731, 80.7718]}
        zoom={7}
        className="w-full h-full"
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {geoJsonData && (
          <GeoJSON
            key={city}
            data={geoJsonData}
            style={getCityStyle}
            onEachFeature={onEachCity}
          />
        )}
      </MapContainer>

      {/* Legend */}
      {!loading && cityFeature && (
        <div style={{ position: 'absolute', bottom: '16px', right: '16px', background: 'rgba(17,24,39,0.92)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.18)', padding: '12px', borderRadius: '10px', zIndex: 1000 }}>
          <h4 style={{ fontSize: '13px', fontWeight: 700, color: '#d1d5db', marginBottom: '8px' }}>{city}</h4>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <div style={{ width: '14px', height: '14px', borderRadius: '4px', backgroundColor: getRiskColor(riskLevel) }} />
            <span style={{ fontSize: '12px', fontWeight: 600, color: '#d1d5db' }}>{riskLevel} Risk</span>
          </div>
          <p style={{ fontSize: '12px', color: '#9ca3af', margin: 0 }}>Score: {Math.round(riskScore * 100)}%</p>
        </div>
      )}
    </div>
  );
}
