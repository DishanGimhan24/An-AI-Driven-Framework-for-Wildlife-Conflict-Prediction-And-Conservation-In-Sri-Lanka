import 'leaflet/dist/leaflet.css';
import { useEffect, useState } from 'react';
import { GeoJSON, MapContainer, TileLayer } from 'react-leaflet';

export default function RiskHeatmap({ districtData, viewMode = 'district', selectedDistrict = '' }) {
  const [geoJsonData, setGeoJsonData] = useState(null);
  const [selectedArea, setSelectedArea] = useState(null);
  const [loading, setLoading] = useState(true);

  // Load appropriate GeoJSON based on view mode
  useEffect(() => {
    loadGeoJSON();
  }, [viewMode]);

  const loadGeoJSON = async () => {
    setLoading(true);
    try {
      // Load district-level or city-level GeoJSON from public folder
      const fileName = viewMode === 'city' ? 'gadm41_LKA_2.json' : 'gadm41_LKA_1.json';
      const response = await fetch(`/data/${fileName}`);
      const data = await response.json();
      setGeoJsonData(data);
    } catch (error) {
      console.error('Failed to load GeoJSON:', error);
    } finally {
      setLoading(false);
    }
  };

  // Get risk color based on level
  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'HIGH':
        return '#EF4444';
      case 'MEDIUM':
        return '#F59E0B';
      case 'LOW':
        return '#10B981';
      default:
        return '#9CA3AF';
    }
  };

  // Get risk data for district or city
  const getAreaRisk = (areaName) => {
    if (!districtData) return null;

    const searchName = areaName.toLowerCase().trim();
    
    // Search in districts array (for district view)
    if (districtData.districts) {
      const district = districtData.districts.find(d => {
        if (!d?.district) return false;
        const dName = d.district.toString().toLowerCase();
        return dName === searchName || searchName.includes(dName) || dName.includes(searchName);
      });
      if (district) return district;
    }

    // Search in cities array (for city view)
    if (districtData.cities) {
      const city = districtData.cities.find(c => {
        if (!c?.name) return false;
        const cName = c.name.toString().toLowerCase();
        return cName === searchName || searchName.includes(cName) || cName.includes(searchName);
      });
      if (city) return city;
    }

    return null;
  };

  // Style each area based on risk level
  const getAreaStyle = (feature) => {
    // Get area name based on view mode
    const areaName = viewMode === 'city' 
      ? feature.properties.NAME_2  // City name from GADM level 2
      : feature.properties.NAME_1; // District name from GADM level 1
    
    const areaRisk = getAreaRisk(areaName);
    const riskLevel = areaRisk?.risk_level || 'LOW';
    const color = getRiskColor(riskLevel);
    
    return {
      fillColor: color,
      fillOpacity: 0.6,
      color: '#333',
      weight: 1.5,
      opacity: 0.8
    };
  };

  // Handle area interactions
  const onEachArea = (feature, layer) => {
    const areaName = viewMode === 'city' 
      ? feature.properties.NAME_2 
      : feature.properties.NAME_1;
    
    const areaRisk = getAreaRisk(areaName);
    
    // Popup content
    const popupContent = `
      <div style="padding: 8px;">
        <h3 style="margin: 0 0 8px 0; font-weight: 600; font-size: 14px;">${areaName}</h3>
        ${areaRisk ? `
          <div style="margin-bottom: 4px;">
            <span style="font-size: 12px; color: #666;">Risk Level:</span>
            <span style="font-weight: 600; margin-left: 4px; color: ${getRiskColor(areaRisk.risk_level)};">
              ${areaRisk.risk_level}
            </span>
          </div>
          <div style="margin-bottom: 4px;">
            <span style="font-size: 12px; color: #666;">Risk Score:</span>
            <span style="font-weight: 600; margin-left: 4px;">
              ${Math.round(areaRisk.risk_score * 100)}%
            </span>
          </div>
        ` : '<div style="font-size: 12px; color: #999;">No data available</div>'}
      </div>
    `;
    
    layer.bindPopup(popupContent);
    
    // Hover effects
    layer.on({
      mouseover: (e) => {
        const layer = e.target;
        layer.setStyle({
          weight: 3,
          fillOpacity: 0.8
        });
        setSelectedArea(areaName);
      },
      mouseout: (e) => {
        const layer = e.target;
        layer.setStyle({
          weight: 1.5,
          fillOpacity: 0.6
        });
        setSelectedArea(null);
      }
    });
  };

  // Filter GeoJSON features for city view (only show selected district's cities)
  const getFilteredGeoJSON = () => {
    if (!geoJsonData) return null;
    
    // For city view, filter by selected district
    if (viewMode === 'city' && selectedDistrict) {
      return {
        ...geoJsonData,
        features: geoJsonData.features.filter(
          feature => feature.properties.NAME_1 === selectedDistrict
        )
      };
    }
    
    return geoJsonData;
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '500px' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ width: '48px', height: '48px', border: '4px solid rgba(255,255,255,0.1)', borderTopColor: 'var(--emerald-500)', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 16px' }} />
          <p style={{ color: '#6b7280', fontSize: '14px' }}>Loading map...</p>
        </div>
      </div>
    );
  }

  const filteredData = getFilteredGeoJSON();

  return (
    <div style={{ position: 'relative', overflow: 'hidden', borderRadius: '12px', height: '500px' }}>
      <MapContainer 
        center={[7.8731, 80.7718]} 
        zoom={viewMode === 'city' ? 10 : 8}
        className="w-full h-full"
        scrollWheelZoom={true}
        key={viewMode}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {filteredData && (
          <GeoJSON
            key={`${viewMode}-${selectedDistrict}-${districtData?.date || ''}-${(districtData?.districts || districtData?.cities || []).map(a => `${a.district || a.name}:${a.risk_score}`).join('|')}`}
            data={filteredData}
            style={getAreaStyle}
            onEachFeature={onEachArea}
          />
        )}
      </MapContainer>

      {/* Legend */}
      <div style={{ position: 'absolute', bottom: '16px', right: '16px', background: 'rgba(17,24,39,0.92)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.18)', padding: '12px', borderRadius: '10px', zIndex: 1000 }}>
        <h4 style={{ fontSize: '12px', fontWeight: 700, color: '#d1d5db', marginBottom: '8px' }}>Risk Level</h4>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {[['#EF4444', 'High Risk'], ['#F59E0B', 'Medium Risk'], ['#10B981', 'Low Risk']].map(([color, label]) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '14px', height: '14px', borderRadius: '4px', backgroundColor: color }} />
              <span style={{ fontSize: '12px', color: '#9ca3af' }}>{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Summary Stats */}
      {districtData?.summary && (
        <div style={{ position: 'absolute', top: '16px', right: '16px', background: 'rgba(17,24,39,0.92)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.18)', padding: '12px', borderRadius: '10px', zIndex: 1000, maxWidth: '180px' }}>
          <h4 style={{ fontSize: '11px', fontWeight: 700, color: '#d1d5db', marginBottom: '8px' }}>
            {viewMode === 'city' ? 'City Summary' : 'District Summary'}
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '12px' }}>
            {[['#ef4444', 'High Risk:', 'high_risk_cities', 'high_risk_districts'], ['#f59e0b', 'Medium Risk:', 'medium_risk_cities', 'medium_risk_districts'], ['#10b981', 'Low Risk:', 'low_risk_cities', 'low_risk_districts']].map(([color, label, k1, k2]) => (
              <div key={label} style={{ display: 'flex', justifyContent: 'space-between', gap: '12px' }}>
                <span style={{ color }}>{label}</span>
                <span style={{ fontWeight: 700, color: '#d1d5db' }}>{districtData.summary[k1] || districtData.summary[k2] || 0}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Hover tooltip */}
      {selectedArea && (
        <div style={{ position: 'absolute', top: '16px', left: '16px', background: 'rgba(17,24,39,0.92)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.18)', padding: '8px 14px', borderRadius: '10px', zIndex: 1000 }}>
          <p style={{ fontSize: '13px', fontWeight: 600, color: '#d1d5db', margin: 0 }}>{selectedArea}</p>
        </div>
      )}
    </div>
  );
}