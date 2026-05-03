import apiClient from './axiosConfig';

export const getStressSummary = async (date = null, regenerate = false) => {
  const params = {};
  if (date) params.date = date;
  if (regenerate) params.regenerate = 'true';
  const response = await apiClient.get('/environmental-stress/summary', { params });
  return response.data.data;
};

export const getSeasonalProfile = async (district, regenerate = false) => {
  const params = { district };
  if (regenerate) params.regenerate = 'true';
  const response = await apiClient.get('/environmental-stress/seasonal-profile', { params });
  return response.data.data;
};

export const getStressCorrelation = async () => {
  const response = await apiClient.get('/environmental-stress/stress-correlation');
  return response.data.data;
};
